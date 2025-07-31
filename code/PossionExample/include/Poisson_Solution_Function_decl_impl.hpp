#pragma once

#include <PanzerAdaptersSTK_config.hpp>

#include <Phalanx_config.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_FieldManager.hpp>

#include <Panzer_Dimension.hpp>
#include <Panzer_FieldLibrary.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_Traits.hpp>
#include <Panzer_Workset.hpp>
#include <Panzer_Workset_Utilities.hpp>
#include <Panzer_BasisIRLayout.hpp>

#include <string>
#include <cmath>

namespace PoissonEquation
{

    using panzer::Cell;
    using panzer::Dim;
    using panzer::Point;

    /** A source for the curl Laplacian that results in the solution
     */
    template <typename EvalT, typename Traits>
    class SimpleSolution : public panzer::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>
    {

    public:
        SimpleSolution(const std::string &name,
                       const panzer::IntegrationRule &ir);

        void postRegistrationSetup(typename Traits::SetupData d,
                                   PHX::FieldManager<Traits> &fm);

        void evaluateFields(typename Traits::EvalData d);

    private:
        typedef typename EvalT::ScalarT ScalarT;

        PHX::MDField<ScalarT, Cell, Point> solution;
        PHX::MDField<ScalarT, Cell, Point, Dim> solution_grad;
        int ir_degree, ir_index;
    };

    // 初始化，和Source基本一致
    template <typename EvalT, typename Traits>
    SimpleSolution<EvalT, Traits>::SimpleSolution(const std::string &name,
                                                  const panzer::IntegrationRule &ir)
    {
        using Teuchos::RCP;

        Teuchos::RCP<PHX::DataLayout> data_layout_scalar = ir.dl_scalar;
        Teuchos::RCP<PHX::DataLayout> data_layout_vector = ir.dl_vector;
        ir_degree = ir.cubature_degree;

        solution = PHX::MDField<ScalarT, Cell, Point>(name, data_layout_scalar);
        solution_grad = PHX::MDField<ScalarT, Cell, Point, Dim>("GRAD_" + name, data_layout_vector);

        this->addEvaluatedField(solution);
        this->addEvaluatedField(solution_grad);

        std::string n = "Simple Solution";
        this->setName(n);
    }

    //**********************************************************************
    template <typename EvalT, typename Traits>
    void SimpleSolution<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData sd,
                                                              PHX::FieldManager<Traits> & /* fm */)
    {
        ir_index = panzer::getIntegrationRuleIndex(ir_degree, (*sd.worksets_)[0], this->wda);
    }

    //**********************************************************************
    template <typename EvalT, typename Traits>
    void SimpleSolution<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
    {
        using panzer::index_t;
        auto ip_coordinates = this->wda(workset).int_rules[ir_index]->ip_coordinates.get_static_view();
        auto solution_v = solution.get_static_view();
        auto solution_grad_v = solution_grad.get_static_view();

        Kokkos::parallel_for("SimpleSolution",workset.num_cells, 
        KOKKOS_LAMBDA(const index_t cell) {
            for (int point = 0; point < solution_v.extent_int(1); ++point) 
            {
                const double & x = ip_coordinates(cell,point,0);
                const double & y = ip_coordinates(cell,point,1);
                solution_v(cell,point) = std::sin(2*M_PI*x)*std::sin(2*M_PI*y);
                solution_grad_v(cell,point,0) = 2.0*M_PI*std::cos(2*M_PI*x)*std::sin(2*M_PI*y);
                solution_grad_v(cell,point,1) = 2.0*M_PI*std::sin(2*M_PI*x)*std::cos(2*M_PI*y);
            } 
        });
        Kokkos::fence();
    }
}