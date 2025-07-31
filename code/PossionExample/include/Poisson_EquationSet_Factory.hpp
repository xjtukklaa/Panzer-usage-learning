#pragma once

#include <Panzer_EquationSet_Factory.hpp>
#include <Panzer_EquationSet_Factory_Defines.hpp>
#include <Panzer_CellData.hpp>

#include "Poisson_EquationSet_decl_impl.hpp"

namespace PoissonEquation
{
    // 类名加上模板的一个结构
    // 用于定义不同计算模型下的这个类的创建方法
    PANZER_DECLARE_EQSET_TEMPLATE_BUILDER(PoissonEquationSet,PoissonEquationSet)

    class EquationSetFactory : public panzer::EquationSetFactory
    {
        public :
        Teuchos::RCP<panzer::EquationSet_TemplateManager<panzer::Traits>>
        buildEquationSet(const Teuchos::RCP<Teuchos::ParameterList> &params,
                         const int &default_integration_order,
                         const panzer::CellData &cell_data,
                         const Teuchos::RCP<panzer::GlobalData> &global_data,
                         const bool build_transient_support) const
        {
            Teuchos::RCP<panzer::EquationSet_TemplateManager<panzer::Traits>> eq_set =
                Teuchos::rcp(new panzer::EquationSet_TemplateManager<panzer::Traits>);

            bool found = false;

            PANZER_BUILD_EQSET_OBJECTS("Poisson", PoissonEquationSet)

            if (!found)
            {
                std::string msg = "Error - the \"Equation Set\" called \"" + 
                params->get<std::string>("Type") + 
                "\" is not a valid equation set identifier. Please supply the correct factory.\n";
                TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, msg);
            }

            return eq_set;
        }
    };
}