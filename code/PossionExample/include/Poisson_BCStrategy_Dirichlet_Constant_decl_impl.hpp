#pragma once

#include <vector>
#include <string>

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Assert.hpp>

#include <Panzer_BCStrategy_Dirichlet_DefaultImpl.hpp>
#include <Panzer_BCStrategy_Neumann_DefaultImpl.hpp>
#include <Panzer_Traits.hpp>
#include <Panzer_PureBasis.hpp>
#include <Panzer_Constant.hpp>
#include <Panzer_ConstantFlux.hpp>

#include <Phalanx_FieldManager.hpp>
#include <Phalanx_DataLayout_MDALayout.hpp>
#include <Phalanx_MDField.hpp>
#include <Phalanx_DataLayout.hpp>

namespace PoissonEquation
{
    template <typename EvalT>
    class BCStrategy_Dirichlet_Constant :
     public panzer::BCStrategy_Dirichlet_DefaultImpl<EvalT>
    {
        public :
            BCStrategy_Dirichlet_Constant(const panzer::BC &bc,
                                          const Teuchos::RCP<panzer::GlobalData> &global_data);
        
            void setup(const panzer::PhysicsBlock &side_pb,
                       const Teuchos::ParameterList &user_data);

            void buildAndRegisterEvaluators(PHX::FieldManager<panzer::Traits> &fm,
                                            const panzer::PhysicsBlock &pb,
                                            const panzer::ClosureModelFactory_TemplateManager<panzer::Traits> &factory,
                                            const Teuchos::ParameterList &models,
                                            const Teuchos::ParameterList &user_data) const;
            // 计算的模型名称
            std::string residual_name;
            Teuchos::RCP<panzer::PureBasis> basis;
    };

    // 初始化这个类，传入参数用于初始化基类
    template <typename EvalT>
    BCStrategy_Dirichlet_Constant<EvalT>::
        BCStrategy_Dirichlet_Constant(const panzer::BC &bc, 
                    const Teuchos::RCP<panzer::GlobalData> &global_data)
        : panzer::BCStrategy_Dirichlet_DefaultImpl<EvalT>(bc, global_data)
    {
        // 检查边界是否符合要求
        TEUCHOS_ASSERT(this->m_bc.strategy() == "Constant");
    }

    template <typename EvalT>
    void BCStrategy_Dirichlet_Constant<EvalT>::
        setup(const panzer::PhysicsBlock &side_pb,
              const Teuchos::ParameterList & /* user_data */)
    {
        using std::pair;
        using std::string;
        using std::vector;
        using Teuchos::RCP;
        typedef vector<pair<string, RCP<panzer::PureBasis>>> DofPureBasisMap;

        // equationSetName就是对应的自由度名称
        this->required_dof_names.push_back(this->m_bc.equationSetName());
        // 需要添加约束的计算模型
        this->residual_name = "Residual_" + this->m_bc.identifier();
        // 创建计算模型和自由度直接的关系
        this->residual_to_dof_names_map[residual_name] = this->m_bc.equationSetName();
        // 创建计算模型和对应物理场的关系
        this->residual_to_target_field_map[residual_name] = 
                       "Constant_" + this->m_bc.equationSetName();
        // 找到当前自由度对应的基函数
        const DofPureBasisMap &dofs = side_pb.getProvidedDOFs();

        for (DofPureBasisMap::const_iterator dof_it = dofs.begin();
            dof_it != dofs.end(); ++dof_it)
        {
            if (dof_it->first == this->m_bc.equationSetName())
                this->basis = dof_it->second;
        }
        // 如果没找到则报错
        TEUCHOS_TEST_FOR_EXCEPTION(Teuchos::is_null(this->basis), 
                                   std::runtime_error,
            "Error the name \"" << this->m_bc.equationSetName()
            << "\" is not a valid DOF for the boundary condition:\n"
            << this->m_bc << "\n");
    }

    template <typename EvalT>
    void BCStrategy_Dirichlet_Constant<EvalT>::
        buildAndRegisterEvaluators(PHX::FieldManager<panzer::Traits> &fm,
                                const panzer::PhysicsBlock & /* pb */,
                                const panzer::ClosureModelFactory_TemplateManager<panzer::Traits> & /* factory */,
                                const Teuchos::ParameterList & /* models */,
                                const Teuchos::ParameterList & /* user_data */) const
    {
        using Teuchos::ParameterList;
        using Teuchos::RCP;
        using Teuchos::rcp;

        // 网格信息
        // Teuchos::RCP<PHX::DataLayout> cell_data;
        // 形函数的值
        // Teuchos::RCP<PHX::DataLayout> functional;
        // 形函数的梯度
        // Teuchos::RCP<PHX::DataLayout> functional_grad;
        // 形函数的二阶梯度
        // Teuchos::RCP<PHX::DataLayout> functional_D2;
        // 单元上的坐标
        // Teuchos::RCP<PHX::DataLayout> coordinates;
        // 单元矩阵
        // Teuchos::RCP<PHX::DataLayout> local_mat_layout;

        // 项物理场中添加一个常值场
        {
            ParameterList p("BC Constant Dirichlet");
            p.set("Name", "Constant_" + this->m_bc.equationSetName());
            p.set("Data Layout", basis->functional);
            p.set("Value", this->m_bc.params()->template get<double>("Value"));

            RCP<PHX::Evaluator<panzer::Traits>> op =
                rcp(new panzer::Constant<EvalT, panzer::Traits>(p));

            this->template registerEvaluator<EvalT>(fm, op);
        }
    }
}