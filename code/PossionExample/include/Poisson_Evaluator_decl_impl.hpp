#pragma once
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardParameterEntryValidators.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Assert.hpp"
#include "Phalanx_DataLayout_MDALayout.hpp"
#include "Phalanx_FieldManager.hpp"

#include "Panzer_EquationSet_DefaultImpl.hpp"
#include "Panzer_Traits.hpp"

#include "Panzer_IntegrationRule.hpp"
#include "Panzer_BasisIRLayout.hpp"
#include "Panzer_Workset.hpp"
#include "Panzer_Workset_Utilities.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

#include <string>
// include evaluators here
#include "Panzer_Integrator_BasisTimesScalar.hpp"
#include "Panzer_Integrator_TransientBasisTimesScalar.hpp"
#include "Panzer_Integrator_BasisTimesVector.hpp"
#include "Panzer_Integrator_BasisTimesTensorTimesVector.hpp"
#include "Panzer_Product.hpp"

#include "Panzer_EvaluatorStyle.hpp"
#include "Panzer_Evaluator_WithBaseImpl.hpp"

// Phalanx
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Panzer_HierarchicParallelism.hpp"

using panzer::BASIS;
using panzer::Cell;
using panzer::Dim;
using panzer::Point;

namespace PoissonEquation
{
    template <typename EvalT, typename Traits>
    class Residual_Laplace : public panzer::EvaluatorWithBaseImpl<Traits>,
                             public PHX::EvaluatorDerived<EvalT, Traits>
    {
    public:
        Residual_Laplace(
            const Teuchos::ParameterList &p,
            const panzer::EvaluatorStyle &evalStyle,
            const std::string &resName,
            const std::string &gradName,
            const panzer::BasisIRLayout &basis,
            const panzer::IntegrationRule &ir);

        void postRegistrationSetup(typename Traits::SetupData workset,
                                   PHX::FieldManager<Traits> &fm);

        void evaluateFields(typename Traits::EvalData workset);
        //
        KOKKOS_INLINE_FUNCTION
        void operator()(const Kokkos::TeamPolicy<PHX::exec_space>::member_type &team) const;

    private:
        /**
         *  \brief The scalar type.
         */
        using ScalarT = typename EvalT::ScalarT;

        using scratch_view = Kokkos::View<ScalarT *,
                                          typename PHX::DevLayout<ScalarT>::type,
                                          typename PHX::exec_space::scratch_memory_space,
                                          Kokkos::MemoryUnmanaged>;

        const panzer::EvaluatorStyle evalStyle_;

        std::string basisName_;

        std::size_t basisIndex_;

        // 梯度向量
        PHX::MDField<const ScalarT, Cell, Point, Dim> grad_;

        // 计算场
        PHX::MDField<ScalarT, Cell, BASIS> field_;

        // 基函数的值
        PHX::MDField<double, Cell, BASIS, Point, Dim> basis_;

        // 导热率
        double kappa;

        // 用于提取基函数的值
        int irDegree_, irIndex_;
    };

    template <typename EvalT, typename Traits>
    Residual_Laplace<EvalT, Traits>::Residual_Laplace(
        const Teuchos::ParameterList &p,
        const panzer::EvaluatorStyle &evalStyle,
        const std::string &resName,
        const std::string &gradName,
        const panzer::BasisIRLayout &basis,
        const panzer::IntegrationRule &ir)
        : evalStyle_(evalStyle),
          basisName_(basis.name())
    {
        kappa = p.get<double>("Heat Conduction");

        irDegree_ = ir.cubature_degree;

        field_ = PHX::MDField<ScalarT, Cell, BASIS>(resName, basis.functional);

        if (evalStyle == panzer::EvaluatorStyle::CONTRIBUTES)
            this->addContributedField(field_);
        else
            this->addEvaluatedField(field_);

        grad_ = PHX::MDField<const ScalarT, Cell, Point, Dim>(gradName, ir.dl_vector);
        this->addDependentField(grad_);

        std::string n("Integrator_WaveFunction (");
        if (evalStyle == panzer::EvaluatorStyle::CONTRIBUTES)
            n += "CONTRIBUTES";
        else
            n += "EVALUATES";

        n += "):  " + field_.fieldTag().name();
        this->setName(n);
    };

    template <typename EvalT, typename Traits>
    void Residual_Laplace<EvalT, Traits>::
        postRegistrationSetup(typename Traits::SetupData workset,
                              PHX::FieldManager<Traits> &fm)
    {
        basisIndex_ = panzer::getBasisIndex(basisName_, (*workset.worksets_)[0], this->wda);
        irIndex_ = panzer::getIntegrationRuleIndex(irDegree_, (*workset.worksets_)[0], this->wda);
    };
    ////////////////////
    template <typename EvalT, typename Traits>
    KOKKOS_INLINE_FUNCTION void Residual_Laplace<EvalT, Traits>::
    operator()(const Kokkos::TeamPolicy<PHX::exec_space>::member_type &team) const
    {
        using panzer::EvaluatorStyle;
        const int cell = team.league_rank();

        const int numQP(grad_.extent(1)),
            numDim(grad_.extent(2)),
            numBases(basis_.extent(1));

        if (evalStyle_ == EvaluatorStyle::EVALUATES)
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, 0, numBases), 
                [&](const int basis){ 
                    field_(cell, basis) = 0.0; 
            });

        for (int qp(0); qp < numQP; ++qp)
        {
            for (int dim(0); dim < numDim; ++dim)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, numBases), 
                    [&](const int basis){ 
                        field_(cell, basis) += basis_(cell, basis, qp, dim) * 
                                               kappa * grad_(cell, qp, dim); 
                });
            }
        }
    }

    template <typename EvalT, typename Traits>
    void Residual_Laplace<EvalT, Traits>::
        evaluateFields(typename Traits::EvalData workset)
    {
        using Kokkos::parallel_for;
        using Kokkos::TeamPolicy;

        basis_ = this->wda(workset).bases[basisIndex_]->weighted_grad_basis;

        auto policy = panzer::HP::inst().teamPolicy<ScalarT, PHX::Device>(workset.num_cells);
        parallel_for(this->getName(), policy, *this);
    };
}
