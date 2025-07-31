#pragma once

#include <string>
#include "Sacado_mpl_apply.hpp"
#include "Teuchos_RCP.hpp"
#include "Poisson_ClosureModel_Factory_decl_impl.hpp"

namespace PoissonEquation
{

    class ClosureModelFactory_TemplateBuilder
    {
    public:
        template <typename EvalT>
        Teuchos::RCP<panzer::ClosureModelFactoryBase> build() const
        {
            return Teuchos::rcp(static_cast<panzer::ClosureModelFactoryBase *>
                    (new PoissonEquation::ModelFactory<EvalT>));
        }
    };

}