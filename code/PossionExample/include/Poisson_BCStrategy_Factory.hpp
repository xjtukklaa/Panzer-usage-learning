#pragma once

#include <Teuchos_RCP.hpp>
#include <Panzer_Traits.hpp>
#include <Panzer_BCStrategy_Factory.hpp>
#include <Panzer_BCStrategy_TemplateManager.hpp>
#include <Panzer_BCStrategy_Factory_Defines.hpp>

#include "Poisson_BCStrategy_Dirichlet_Constant_decl_impl.hpp"

namespace PoissonEquation
{
    // 这个宏相当神奇，可能是之前写代码写的少了
    // 通过宏定义对应类的一个BCStrategy_Dirichlet_Constant_TemplateBuilder结构
    /*
        struct BCStrategy_Dirichlet_Constant_TemplateBuilder{...}
    */ 
    PANZER_DECLARE_BCSTRATEGY_TEMPLATE_BUILDER(BCStrategy_Dirichlet_Constant,
                                               BCStrategy_Dirichlet_Constant)

    struct BCStrategyFactory : public panzer::BCStrategyFactory
    {

        Teuchos::RCP<panzer::BCStrategy_TemplateManager<panzer::Traits>>
        buildBCStrategy(const panzer::BC &bc, const Teuchos::RCP<panzer::GlobalData> &global_data) const
        {

            Teuchos::RCP<panzer::BCStrategy_TemplateManager<panzer::Traits>> bcs_tm =
                Teuchos::rcp(new panzer::BCStrategy_TemplateManager<panzer::Traits>);

            bool found = false;

            PANZER_BUILD_BCSTRATEGY_OBJECTS("Constant",
                                            BCStrategy_Dirichlet_Constant)

            TEUCHOS_TEST_FOR_EXCEPTION(!found, std::logic_error,
                                       "Error - the BC Strategy called \"" 
                                       << bc.strategy() 
                                       << "\" is not a valid identifier in the BCStrategyFactory.  Either add a "
                                          "valid implementation to your factory or fix your input file.  The "
                                          "relevant boundary condition is:\n\n"
                                        << bc << std::endl);

            return bcs_tm;
        }
    };
}