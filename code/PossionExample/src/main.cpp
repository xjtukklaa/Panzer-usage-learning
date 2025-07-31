#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_StackedTimer.hpp>
#include <Teuchos_FancyOStream.hpp>

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_GlobalMPISession.hpp>

#include <PanzerAdaptersSTK_config.hpp>
#include <Panzer_GlobalData.hpp>
#include <Panzer_Workset_Builder.hpp>
#include <Panzer_WorksetContainer.hpp>
#include <Panzer_AssemblyEngine.hpp>
#include <Panzer_AssemblyEngine_InArgs.hpp>
#include <Panzer_AssemblyEngine_TemplateManager.hpp>
#include <Panzer_AssemblyEngine_TemplateBuilder.hpp>
#include <Panzer_LinearObjFactory.hpp>
#include <Panzer_TpetraLinearObjFactory.hpp>
#include <Panzer_DOFManagerFactory.hpp>
#include <Panzer_FieldManagerBuilder.hpp>
#include <Panzer_PureBasis.hpp>
#include <Panzer_GlobalData.hpp>
#include <Panzer_ResponseLibrary.hpp>
#include <Panzer_ResponseEvaluatorFactory_Functional.hpp>
#include <Panzer_Response_Functional.hpp>

#include <PanzerAdaptersSTK_config.hpp>
#include <Panzer_STK_WorksetFactory.hpp>
#include <Panzer_STKConnManager.hpp>
#include <Panzer_STK_Version.hpp>
#include <Panzer_STK_Interface.hpp>
#include <Panzer_STK_SquareQuadMeshFactory.hpp>
#include <Panzer_STK_SquareTriMeshFactory.hpp>
#include <Panzer_STK_ExodusReaderFactory.hpp>
#include <Panzer_STK_SetupUtilities.hpp>
#include <Panzer_STK_Utilities.hpp>
#include <Panzer_STK_ResponseEvaluatorFactory_SolutionWriter.hpp>

#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosMultiVecTraits_Tpetra.hpp>
#include <BelosOperatorTraits_Tpetra.hpp>

#include <MatrixMarket_Tpetra.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MatrixIO.hpp>

#include <Ifpack2_Preconditioner.hpp>
#include <Ifpack2_Factory.hpp>

#include "../include/Poisson_BCStrategy_Factory.hpp"
#include "../include/Poisson_EquationSet_Factory.hpp"
#include "../include/Poisson_ClosureModel_Factory_TemplateBuilder.hpp"

#include <sstream>
#include <fstream>
#include <cmath>

using Teuchos::RCP;
using Teuchos::rcp;

int main(int argc, char *argv[])
{
  using panzer::StrPureBasisComp;
  using panzer::StrPureBasisPair;
  using Teuchos::RCP;
  using Teuchos::rcp_dynamic_cast;

  Tpetra::ScopeGuard MyScope(&argc, &argv);
  {
    RCP<const Teuchos::Comm<int>> tComm = Tpetra::getDefaultComm();
    // 并行下的信息输出
    // 基本等价于std::cout但只在0号节点输出
    Teuchos::FancyOStream out(Teuchos::rcpFromRef(std::cout));
    out.setOutputToRootOnly(0);
    out.setShowProcRank(true);
    // *****************************************************
    // 参数输入处理
    // x,y方向的网格个数，积分阶数
    int x_elements = 50, y_elements = 50, basis_order = 1;
    // 网格类型，三角形和四边形
    std::string celltype = "Quad"; // or "Tri"
    Teuchos::CommandLineProcessor clp;
    clp.setOption("cell", &celltype);
    clp.setOption("x-elements", &x_elements);
    clp.setOption("y-elements", &y_elements);
    clp.setOption("basis-order", &basis_order);
    if (clp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
      return EXIT_FAILURE;
    }
    // *****************************************************
    // 创建求解方程和边界条件处理，两者用于指定网格的物理区域
    RCP<PoissonEquation::EquationSetFactory> eqset_factory = rcp(new PoissonEquation::EquationSetFactory);
    PoissonEquation::BCStrategyFactory bc_factory;
    // 创建网格
    RCP<panzer_stk::STK_MeshFactory> mesh_factory;
    if (celltype == "Quad")
      mesh_factory = Teuchos::rcp(new panzer_stk::SquareQuadMeshFactory);
    else if (celltype == "Tri")
      mesh_factory = Teuchos::rcp(new panzer_stk::SquareTriMeshFactory);
    else
      throw std::runtime_error("not supported celltype argument: try Quad or Tri");
    {
      RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
      pl->set("X Blocks", 1);
      pl->set("Y Blocks", 1);
      pl->set("X Elements", x_elements);
      pl->set("Y Elements", y_elements);
      mesh_factory->setParameterList(pl);
    }
    // workset的大小，指定在单个进程上的计算区域个数
    const std::size_t workset_size = 2000;
    // 创建网格，上面只有网格的基本信息
    RCP<panzer_stk::STK_Interface> mesh = mesh_factory->buildUncommitedMesh(MPI_COMM_WORLD);
    // 创建物理区域，实际上就是将网格块和
    // 求解方程，边界条件联系起来，打包成一个物理模型
    Teuchos::RCP<Teuchos::ParameterList> ipb = Teuchos::parameterList("Physics Blocks");
    std::vector<panzer::BC> bcs;
    // 创建Poisson方程的控制参数,这个参数列表是最顶层的
    {
      const int integration_order = 10;
      Teuchos::ParameterList &p = ipb->sublist("Poisson Physics");
      p.set("Type", "Poisson");
      p.set("Model ID", "solid");
      p.set("Basis Type", "HGrad");
      p.set("Basis Order", basis_order);
      p.set("Integration Order", integration_order);
    }
    // 四个边界的参数列表,边界类型都为Dirichlet,值都为0
    // 左侧边界
    {
      std::size_t bc_id = 0;
      panzer::BCType bctype = panzer::BCT_Dirichlet;
      std::string sideset_id = "left";
      std::string element_block_id = "eblock-0_0";
      std::string dof_name = "TEMPERATURE";
      std::string strategy = "Constant";
      double value = 0.0;
      Teuchos::ParameterList p;
      p.set("Value", value);
      panzer::BC bc(bc_id, bctype, sideset_id, element_block_id, dof_name,
                    strategy, p);
      bcs.push_back(bc);
    }
    // 上侧边界
    {
      std::size_t bc_id = 1;
      panzer::BCType bctype = panzer::BCT_Dirichlet;
      std::string sideset_id = "top";
      std::string element_block_id = "eblock-0_0";
      std::string dof_name = "TEMPERATURE";
      std::string strategy = "Constant";
      double value = 0.0;
      Teuchos::ParameterList p;
      p.set("Value", value);
      panzer::BC bc(bc_id, bctype, sideset_id, element_block_id, dof_name,
                    strategy, p);
      bcs.push_back(bc);
    }
    // 右侧边界
    {
      std::size_t bc_id = 2;
      panzer::BCType bctype = panzer::BCT_Dirichlet;
      std::string sideset_id = "right";
      std::string element_block_id = "eblock-0_0";
      std::string dof_name = "TEMPERATURE";
      std::string strategy = "Constant";
      double value = 0.0;
      Teuchos::ParameterList p;
      p.set("Value", value);
      panzer::BC bc(bc_id, bctype, sideset_id, element_block_id, dof_name,
                    strategy, p);
      bcs.push_back(bc);
    }
    // 下侧边界
    {
      std::size_t bc_id = 3;
      panzer::BCType bctype = panzer::BCT_Dirichlet;
      std::string sideset_id = "bottom";
      std::string element_block_id = "eblock-0_0";
      std::string dof_name = "TEMPERATURE";
      std::string strategy = "Constant";
      double value = 0.0;
      Teuchos::ParameterList p;
      p.set("Value", value);
      panzer::BC bc(bc_id, bctype, sideset_id, element_block_id, dof_name,
                    strategy, p);
      bcs.push_back(bc);
    }
    // 创建物理分区
    std::vector<RCP<panzer::PhysicsBlock>> physicsBlocks;
    {
      // 是否计算瞬态项
      bool build_transient_support = false;
      // 这里其实根据worksetsize和网格使用zoltan/zoltan2进行负载均衡分区
      const panzer::CellData volume_cell_data(workset_size, mesh->getCellTopology("eblock-0_0"));
      // 主要声明一个输出，不知所谓()
      Teuchos::RCP<panzer::GlobalData> gd = panzer::createGlobalData();
      // 默认的积分阶数，
      int default_integration_order = 1;
      // 将网格，求解方程，边界条件打包为一个物理区域
      // 这个设计思路和正常使用Comsol有点像
      // 创建网格后,选择求解方程和边界条件 -> PhysicsBlock
      // 再添加源项和后处理的误差分析结果  -> ClosureModel
      RCP<panzer::PhysicsBlock> pb = rcp(new panzer::PhysicsBlock(ipb,
                                                                  "eblock-0_0",
                                                                  default_integration_order,
                                                                  volume_cell_data,
                                                                  eqset_factory,
                                                                  gd,
                                                                  build_transient_support));
      // 按照这个代码的说法,同组网格是可以耦合多个不同的方程和边界条件
      // 比如流场和热场,力场和电磁场之类的,因为封装的比较高层
      // 耦合场的自由度,单元类型由内部处理
      // HGrad + RT + HCurl + HGrad^Dim - Some Magic -> 我说棱镜开局特别强有人懂吗
      physicsBlocks.push_back(pb);
    }
    // 下一步是将物理分区和自由度联系起来
    {
      // 这就一个Poisson问题,只有一个物理分区
      RCP<panzer::PhysicsBlock> pb = physicsBlocks[0];
      // StrPureBasisPair = map<DofName, Basis>
      // 获得自由度名称,就是在eqset_factory里面声明的自由度,"TEMPERATURE"
      const std::vector<StrPureBasisPair> &blockFields = pb->getProvidedDOFs();
      // 考虑到可能有多个物理分区
      // 开一个迭代容器
      std::set<StrPureBasisPair, StrPureBasisComp> fieldNames;
      fieldNames.insert(blockFields.begin(), blockFields.end());
      // 网格上添加自由度
      // "eblock-0_0"就是elementBlockID
      std::set<StrPureBasisPair, StrPureBasisComp>::const_iterator fieldItr;
      for (fieldItr = fieldNames.begin(); fieldItr != fieldNames.end(); ++fieldItr)
        mesh->addSolutionField(fieldItr->first, pb->elementBlockID());
      // 最后完成网格创建
      mesh_factory->completeMeshConstruction(*mesh, MPI_COMM_WORLD);
    }
    // *****************************************************
    // 创建自由度管理器,线性问题
    // conn_manager从网格中创建,主要管理网格之间的拓扑关系
    const Teuchos::RCP<panzer::ConnManager> conn_manager =
        Teuchos::rcp(new panzer_stk::STKConnManager(mesh));
    // 创建当前physicsBlocks上的自由度管理
    // 名称就叫做"TEMPERATURE"
    panzer::DOFManagerFactory globalIndexerFactory;
    RCP<panzer::GlobalIndexer> dofManager = 
             globalIndexerFactory.buildGlobalIndexer(
                  Teuchos::opaqueWrapper(MPI_COMM_WORLD), physicsBlocks, conn_manager);
    out<<"Number Of Dofs : "<<dofManager->getNumOwnedAndGhosted()<<std::endl;          
    // 创建Tpetra线性问题,A表示矩阵,X表示解向量,F表示右手项
    Teuchos::RCP<panzer::LinearObjFactory<panzer::Traits>> linObjFactory = 
        Teuchos::rcp(new panzer::TpetraLinearObjFactory
           <panzer::Traits, double, panzer::LocalOrdinal, panzer::GlobalOrdinal>(tComm, dofManager));
    // *****************************************************
    // 创建workset,这里以我的理解粗浅的解释以下为啥要创建分区
    // 以及将网格打包的好处,实话实说,有限元计算可能只有Jacobian
    // 不同,其他基本一致.故此将多个单元打包在一起,同时计算整组的
    // Jacobian,弱形式完全一致的情况下能算的更快()
    Teuchos::RCP<panzer_stk::WorksetFactory> wkstFactory =
          Teuchos::rcp(new panzer_stk::WorksetFactory(mesh));
    Teuchos::RCP<panzer::WorksetContainer> wkstContainer = 
          Teuchos::rcp(new panzer::WorksetContainer);
      wkstContainer->setFactory(wkstFactory);
      
      for(size_t i=0;i<physicsBlocks.size();i++)
      {
        wkstContainer->setNeeds(physicsBlocks[i]->elementBlockID(),
                 physicsBlocks[i]->getWorksetNeeds());        
      }
      wkstContainer->setWorksetSize(workset_size);
      wkstContainer->setGlobalIndexer(dofManager);
    // *****************************************************
    // 求解结果输入输出,这里输出的时exo文件,Paraview可以直接打开
    // 此处主要输出求解结果
    Teuchos::RCP<panzer::ResponseLibrary<panzer::Traits> > stkIOResponseLibrary
      = Teuchos::rcp(new panzer::ResponseLibrary<panzer::Traits>(wkstContainer,dofManager,linObjFactory));
    {
      // get a vector of all the element blocks
      std::vector<std::string> eBlocks;
      mesh->getElementBlockNames(eBlocks);

      panzer_stk::RespFactorySolnWriter_Builder builder;
      builder.mesh = mesh;
      stkIOResponseLibrary->addResponse("Main Field Output",eBlocks,builder);
    }
    // 和上面类似,但是输出误差结果
    Teuchos::RCP<panzer::ResponseLibrary<panzer::Traits> > exampleResponseLibrary
        = Teuchos::rcp(new panzer::ResponseLibrary<panzer::Traits>(wkstContainer,dofManager,linObjFactory));
    {
      const int integration_order = 10;

      std::vector<std::string> eBlocks;
      mesh->getElementBlockNames(eBlocks);

      panzer::FunctionalResponse_Builder<int,int> builder;
      builder.comm = MPI_COMM_WORLD;
      builder.cubatureDegree = integration_order;
      builder.requiresCellIntegral = true;

      builder.quadPointField = "TEMPERATURE_L2_ERROR";
      exampleResponseLibrary->addResponse("L2 Error",eBlocks,builder);

      builder.quadPointField = "TEMPERATURE_H1_ERROR";
      exampleResponseLibrary->addResponse("H1 Error",eBlocks,builder);

      builder.quadPointField = "AREA";
      exampleResponseLibrary->addResponse("Area",eBlocks,builder);
    }
    // *****************************************************
    // 设定闭合模型,实际上就是将源项,误差分析和上面的物理分区进行打包
    // 相当于后处理和源项部分
    panzer::ClosureModelFactory_TemplateManager<panzer::Traits> cm_factory;
    PoissonEquation::ClosureModelFactory_TemplateBuilder cm_builder;
    cm_factory.buildObjects(cm_builder);

    Teuchos::ParameterList closure_models("Closure Models");
    {
      // 之前那个常值为1的是为了计算面积大小
      closure_models.sublist("solid").sublist("AREA").set<double>("Value",1.0);
      // 添加一个源项
      closure_models.sublist("solid").sublist("SOURCE_TEMPERATURE").set<std::string>("Type","SIMPLE SOURCE");
      // 计算L2范数
      closure_models.sublist("solid").sublist("TEMPERATURE_L2_ERROR").set<std::string>("Type","L2 ERROR_CALC");
      closure_models.sublist("solid").sublist("TEMPERATURE_L2_ERROR").set<std::string>("Field A","TEMPERATURE");
      closure_models.sublist("solid").sublist("TEMPERATURE_L2_ERROR").set<std::string>("Field B","TEMPERATURE_EXACT");
      // 计算H1范数
      closure_models.sublist("solid").sublist("TEMPERATURE_H1_ERROR").set<std::string>("Type","H1 ERROR_CALC");
      closure_models.sublist("solid").sublist("TEMPERATURE_H1_ERROR").set<std::string>("Field A","TEMPERATURE");
      closure_models.sublist("solid").sublist("TEMPERATURE_H1_ERROR").set<std::string>("Field B","TEMPERATURE_EXACT");
      // 将真解添加进去作为u^*
      closure_models.sublist("solid").sublist("TEMPERATURE_EXACT").set<std::string>("Type","TEMPERATURE_EXACT");
    }
    // *****************************************************
    // 求解场管理相当于把所有都打包在一起
    // 啥也没有的用户参数
    Teuchos::ParameterList user_data("User Data"); 
    
    Teuchos::RCP<panzer::FieldManagerBuilder> fmb =
          Teuchos::rcp(new panzer::FieldManagerBuilder);
    fmb->setWorksetContainer(wkstContainer);
    fmb->setupVolumeFieldManagers(physicsBlocks,cm_factory,closure_models,*linObjFactory,user_data);
    fmb->setupBCFieldManagers(bcs,physicsBlocks,*eqset_factory,cm_factory,bc_factory,closure_models,
                              *linObjFactory,user_data);

    // fmb->writeVolumeGraphvizDependencyFiles("Poisson", physicsBlocks);
    // *****************************************************
    // 开始按照设定好的方式进行组装
    panzer::AssemblyEngine_TemplateManager<panzer::Traits> ae_tm;
    panzer::AssemblyEngine_TemplateBuilder builder(fmb,linObjFactory);
    ae_tm.buildObjects(builder);
    {
        user_data.set<int>("Workset Size",workset_size);

        stkIOResponseLibrary->buildResponseEvaluators(physicsBlocks,
                                                      cm_factory,
                                                      closure_models,
                                                      user_data);

        exampleResponseLibrary->buildResponseEvaluators(physicsBlocks,
                                                        cm_factory,
                                                        closure_models,
                                                        user_data);
    }
    // 创建ghost和不带ghost的两个线性问题
    // ghost值的是在分区之后，时常需要计算当前网格的梯度，海斯矩阵
    // 可能会需要边界网格的信息，FVM是一定需要向量单元才能计算
    // 交界面通量
    // ghost区域在远端进程上，一般被设计为不可更改，只有访问权限
    // 
    RCP<panzer::LinearObjContainer> ghostCont = linObjFactory->buildGhostedLinearObjContainer();
    RCP<panzer::LinearObjContainer> container = linObjFactory->buildLinearObjContainer();
    linObjFactory->initializeGhostedContainer(panzer::LinearObjContainer::X |
                                              panzer::LinearObjContainer::F |
                                              panzer::LinearObjContainer::Mat,*ghostCont);
    linObjFactory->initializeContainer(panzer::LinearObjContainer::X |
                                        panzer::LinearObjContainer::F |
                                        panzer::LinearObjContainer::Mat,*container);
    ghostCont->initialize();
    container->initialize();

    panzer::AssemblyEngineInArgs input(ghostCont,container);
    input.alpha = 0;
    input.beta = 1;

    // 这里设定的求解形式为牛顿迭代
    // F'_u \delta u = -F
    // 线性问题变化为  F(\dot{u},u,t) = Au-b
    // 非线性问题变化为F(\dot{u},u,t) = A(u)u-b
    // 时变问题是将时间离散后，将当前时间步上的方程看作是
    // 当前时间上解向量的非线性方程
    // F(\dot{u},u,t) = \dot{u} + Au-b
    // 这个形式是完全统一的

    // 这里就是模板的好处，同时计算F和对应的Jacobian矩阵
    ae_tm.getAsObject<panzer::Traits::Jacobian>()->evaluate(input);
    // *****************************************************
    // 启用Belos进行线性求解
    // 这里不得不说，这里面的常规线性求解器
    // 比如CG,GMRES,Richardson,CG和GMRES的变种等
    // 比我自己写的强的多，比Matlab自带的求解器收敛也快很多
    // 除了常见的求解器，还提供多右端项的块求解器，重用求解器
    // 这些求解器比较不常见，petsc里面我好像没见到(也肯是用的少)
    // 以及GCOR-DR等一些序列加速器
    Teuchos::ParameterList belosList;
    belosList.set("Num Blocks", 1000);
    belosList.set("Block Size", 1);
    belosList.set("Maximum Iterations", 1000);
    belosList.set("Maximum Restarts", 1);
    belosList.set("Convergence Tolerance", 1.0e-9);
    belosList.set("Timer Label", "Belos Init");
    belosList.set( "Verbosity", Belos::Errors + Belos::Warnings +
                    Belos::TimingDetails);
    const int frequency = 1;
    if (frequency > 0)
      belosList.set( "Output Frequency", frequency );

    using ST = double;
    using LO = panzer::LocalOrdinal;
    using GO = panzer::GlobalOrdinal;
    using NT = panzer::TpetraNodeType;
    using OP  = typename Tpetra::Operator<ST,LO,GO,NT>;
    using MV  = typename Tpetra::MultiVector<ST,LO,GO,NT>;
    using MAT = typename Tpetra::CrsMatrix<ST,LO,GO,NT>;
    using Preconditioner = typename Ifpack2::Preconditioner<ST, LO, GO, NT>;

    using TPLOC = panzer::TpetraLinearObjContainer<ST,LO,GO>;
    auto tp_container = rcp_dynamic_cast<TPLOC>(container);
    Belos::LinearProblem<ST,MV,OP> initProblem(tp_container->get_A(), tp_container->get_x(), tp_container->get_f());
    TEUCHOS_ASSERT(initProblem.setProblem());
    // PseudoBlock变种是多右端项求解器，相比Block来说，子空间不是完全正交
    // 求解更加稳健，Block需要的内存空间比较大
    RCP< Belos::SolverManager<ST,MV,OP> > solver
      = rcp(new Belos::PseudoBlockGmresSolMgr<ST,MV,OP>(rcp(&initProblem,false), rcp(&belosList,false)));

    Belos::ReturnType belos_solve_status = solver->solve();
    TEUCHOS_ASSERT(belos_solve_status == Belos::Converged);

    out << "Linear Solver Converged, achieved tol=" << solver->achievedTol() << ", num iters=" << solver->getNumIters() << std::endl;
    // 最后需要注意的是对于线性问题来说，其实计算的是牛顿方程
    // J * e = - (f - J * 0)
    // 所以实际上计算得到的是u=-e
    tp_container->get_x()->scale(-1.0);
    // *****************************************************
    // 将结果写入文件
    {
      panzer::AssemblyEngineInArgs respInput(ghostCont,container);
      respInput.alpha = 0;
      respInput.beta = 1;

      stkIOResponseLibrary->addResponsesToInArgs<panzer::Traits::Residual>(respInput);
      stkIOResponseLibrary->evaluate<panzer::Traits::Residual>(respInput);

      std::ostringstream filename;
      filename << "output_" << celltype << "_p" << basis_order << ".exo";
      mesh->writeToExodus(filename.str());      
    }
    // *****************************************************
    // 计算误差
    panzer::AssemblyEngineInArgs respInput(ghostCont,container);
    respInput.alpha = 0;
    respInput.beta = 1;

    Teuchos::RCP<panzer::ResponseBase> area_resp = exampleResponseLibrary->getResponse<panzer::Traits::Residual>("Area");
    Teuchos::RCP<panzer::Response_Functional<panzer::Traits::Residual> > area_resp_func =
            Teuchos::rcp_dynamic_cast<panzer::Response_Functional<panzer::Traits::Residual> >(area_resp);
    Teuchos::RCP<Thyra::VectorBase<double> > area_respVec = Thyra::createMember(area_resp_func->getVectorSpace());
    area_resp_func->setVector(area_respVec);

    Teuchos::RCP<panzer::ResponseBase> l2_resp = exampleResponseLibrary->getResponse<panzer::Traits::Residual>("L2 Error");
    Teuchos::RCP<panzer::Response_Functional<panzer::Traits::Residual> > l2_resp_func =
            Teuchos::rcp_dynamic_cast<panzer::Response_Functional<panzer::Traits::Residual> >(l2_resp);
    Teuchos::RCP<Thyra::VectorBase<double> > l2_respVec = Thyra::createMember(l2_resp_func->getVectorSpace());
    l2_resp_func->setVector(l2_respVec);

    Teuchos::RCP<panzer::ResponseBase> h1_resp = exampleResponseLibrary->getResponse<panzer::Traits::Residual>("H1 Error");
    Teuchos::RCP<panzer::Response_Functional<panzer::Traits::Residual> > h1_resp_func =
            Teuchos::rcp_dynamic_cast<panzer::Response_Functional<panzer::Traits::Residual> >(h1_resp);
    Teuchos::RCP<Thyra::VectorBase<double> > h1_respVec = Thyra::createMember(h1_resp_func->getVectorSpace());
    h1_resp_func->setVector(h1_respVec);

    exampleResponseLibrary->addResponsesToInArgs<panzer::Traits::Residual>(respInput);
    exampleResponseLibrary->evaluate<panzer::Traits::Residual>(respInput);

    double area_exact = 1.;

    out << "This is the Basis Order" << std::endl;
    out << "Basis Order = " << basis_order << std::endl;
    out << "This is the L2 Error" << std::endl;
    out << "L2 Error = " << sqrt(l2_resp_func->value) << std::endl;
    out << "This is the H1 Error" << std::endl;
    out << "H1 Error = " << sqrt(h1_resp_func->value) << std::endl;
    out << "This is the error in area" << std::endl;
    out << "Area Error = " << std::abs(area_resp_func->value - area_exact) << std::endl;
  }
  return EXIT_SUCCESS;
}