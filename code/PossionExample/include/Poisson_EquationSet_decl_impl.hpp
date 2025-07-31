// 添加头文件检查，防止重复包含头文件
// 一般两种做法，定义宏和pragma once
#pragma once
#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_Assert.hpp>

#include <Panzer_EquationSet_DefaultImpl.hpp>
#include <Panzer_IntegrationRule.hpp>
#include <Panzer_BasisIRLayout.hpp>
// \dot{T} \phi
#include <Panzer_Integrator_BasisTimesScalar.hpp>
// \nable T \nable v
#include <Panzer_Integrator_GradBasisDotVector.hpp>
// Dofs Values -> Basis Interpolates -> QPoints Values
#include <Panzer_TensorToStdVector.hpp>
#include <Panzer_ScalarToVector.hpp>
#include <Panzer_Sum.hpp>
#include <Panzer_Traits.hpp>

#include <Panzer_ScalarToVector.hpp>

#include <Phalanx_FieldManager.hpp>
#include <Phalanx_DataLayout_MDALayout.hpp>

#include "Poisson_Evaluator_decl_impl.hpp"

namespace PoissonEquation
{
    template <typename EvalT>
    class PoissonEquationSet : public panzer::EquationSet_DefaultImpl<EvalT>
    {
        public :
            // Poisson方程的初始化方法
            PoissonEquationSet(const Teuchos::RCP<Teuchos::ParameterList> &params,
                               const int &default_integration_order,
                               const panzer::CellData &cell_data,
                               const Teuchos::RCP<panzer::GlobalData> &global_data,
                               const bool build_transient_support);
            // 核心的弱形式离散
            void buildAndRegisterEquationSetEvaluators(PHX::FieldManager<panzer::Traits> &fm,
                                                       const panzer::FieldLibrary &field_library,
                                                       const Teuchos::ParameterList &user_data) const;
    };

    // 初始化方法
    template <typename EvalT>
    PoissonEquationSet<EvalT>::PoissonEquationSet(const Teuchos::RCP<Teuchos::ParameterList> &params,
                                            const int &default_integration_order,
                                            const panzer::CellData &cell_data,
                                            const Teuchos::RCP<panzer::GlobalData> &global_data,
                                            const bool build_transient_support)
        : panzer::EquationSet_DefaultImpl<EvalT>(params,default_integration_order,cell_data,
                                                 global_data,build_transient_support)
    {
        // 给定一个参数列表，给出默认的参数值，传入的参数列表
        // 和这个默认的进行对比，如果没有参数则添加否则则不管
        Teuchos::ParameterList valid_parameters;
        this->setDefaultValidParameters(valid_parameters);

        valid_parameters.set("Model ID","","Closure model id associated with this equaiton set");
        valid_parameters.set("Basis Type", "HGrad", "Type of Basis to use");
        valid_parameters.set("Basis Order", 1, "Order of the basis");
        valid_parameters.set("Integration Order", -1, "Order of the integration rule");

        params->validateParametersAndSetDefaults(valid_parameters);

        // 获取所需要的参数
        // 基函数的类型，基本上就四类HGrad->HCurl->HDiv->HVol
        // HGrad适合于计算梯度项，在传热问题，力学问题中常见，典型的就是拉格朗日单元
        // HCurl适合于计算旋度项，在电磁问题常见，典型的比如Nédélec单元
        // HDiv 适合于计算散度项，在流体用的多，典型的比如RT压力速度元
        // HVol 有限体积法或有限差分法
        std::string basis_type = params->get<std::string>("Basis Type");
        // 基函数的阶数
        int basis_order = params->get<int>("Basis Order");
        // 积分点的阶数
        int integration_order = params->get<int>("Integration Order");
        // 模型名称，这个模型指的是闭合模型，包括网格，边界和源项
        std::string model_id = params->get<std::string>("Model ID");

        // 创建一个自由度，这个名称就对应这个自由度
        std::string dof_name = "TEMPERATURE";
        // 之前提到过重载模板计算三类模型
        // EvalT == double，计算有限元弱形式方程F(\dot{u},u,t)=0，
        // 默认名称为 "RESIDUAL_" + dof_name
        // EvalT == FADType，计算非线性的Jacobian矩阵
        // EvalT == FADType，进行敏度分析
        this->addDOF(dof_name,basis_type,basis_order,integration_order);

        // 对于Poisson方程，其弱形式只需要计算梯度
        // 梯度的默认名称为"GRAD_" + dof_name
        this->addDOFGrad(dof_name);

        // 判断是否添加瞬态项
        // 时变项的默认名称为"DXDT_" + dof_name
        if (build_transient_support)
            this->addDOFTimeDerivative(dof_name);

        // 生成一个闭合模型
        this->addClosureModel(model_id);

        // 最后初始化自由度
        this->setupDOFs();
    }

    template <typename EvalT>
    void PoissonEquationSet<EvalT>::
        buildAndRegisterEquationSetEvaluators(PHX::FieldManager<panzer::Traits> &fm,
                                              const panzer::FieldLibrary &field_library,
                                              const Teuchos::ParameterList &user_data) const
    {
        using panzer::BasisIRLayout;
        using panzer::EvaluatorStyle;
        using panzer::IntegrationRule;
        using panzer::Integrator_BasisTimesScalar;
        using panzer::Integrator_GradBasisDotVector;
        using panzer::Traits;
        using PHX::Evaluator;
        using std::string;
        using Teuchos::ParameterList;
        using Teuchos::RCP;
        using Teuchos::rcp;

        string dof_name = "TEMPERATURE";
        // 积分类型
        RCP<IntegrationRule> integrationrule = this->getIntRuleForDOF(dof_name);
        // 基函数类型
        RCP<BasisIRLayout> basisir = this->getBasisIRLayoutForDOF(dof_name);

        // 瞬态项计算，\int \dot{T} v
        if (this->buildTransientSupport())
        {
            // resName表示提交到哪一个计算模型中
            // valName表示和试探函数v乘积的另外一部分
            string resName("RESIDUAL_" + dof_name), 
                   valName("DXDT_" + dof_name);
            double multiplier(1.);
            // EvaluatorStyle有两个模式
            // CONTRIBUTES表示计算完加入方程内
            // EVALUATES  表示计算完就结束
            RCP<Evaluator<Traits>> op =
              rcp(new Integrator_BasisTimesScalar<EvalT,Traits>(EvaluatorStyle::CONTRIBUTES,
                                                                resName,valName,
                                                                *basisir,*integrationrule,multiplier));
            // 将这个计算部分添加到FieldManager中
            this->template registerEvaluator<EvalT>(fm,op);
        }

        // 计算扩散项, \int \nable T \cdot \nable v
        {
            // Panzer自带实现
            // Teuchos::ParameterList p("Laplace Term");
            // p.set("Residual Name","RESIDUAL_" + dof_name);
            // p.set("Flux Name","GRAD_" + dof_name);
            // p.set("Basis",basisir);
            // p.set("IR",integrationrule);
            // p.set("Multiplier",1.);

            // RCP<Evaluator<Traits>> op = 
            //  rcp(new Integrator_GradBasisDotVector<EvalT,Traits>(p));
            
            // 自行编写实现
            string resName("RESIDUAL_" + dof_name), 
                   gradName("GRAD_" + dof_name);
            Teuchos::ParameterList p("Laplace Term");
            p.set("Heat Conduction",1.);

            RCP<Evaluator<Traits>> op = 
                rcp(new Residual_Laplace<EvalT,Traits>(p,
                                                        EvaluatorStyle::EVALUATES,
                                                        resName,gradName,
                                                        *basisir,*integrationrule));

            this->template registerEvaluator<EvalT>(fm,op);                                                   
        }

        // 源项, \int Q v
        {
            string resName("RESIDUAL_" + dof_name);
            string valName("SOURCE_" + dof_name);
            double multiplier(-1.);
            RCP<Evaluator<Traits>> op = 
             rcp(new Integrator_BasisTimesScalar<EvalT, Traits>(EvaluatorStyle::CONTRIBUTES,
                                                                resName, valName, 
                                                               *basisir,*integrationrule,multiplier));
            this->template registerEvaluator<EvalT>(fm,op);
        }

        // 上述三个部分构成了一个时变Poisson方程
        // RESIDUAL = \dot{T} v + \nable T \cdot \nable v - Q v = 0
    }
}