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

    template <typename EvalT, typename Traits>
    class SimpleSource : public panzer::EvaluatorWithBaseImpl<Traits>,
                         public PHX::EvaluatorDerived<EvalT, Traits>
    {
        public:
            SimpleSource(const std::string &name,
                        const panzer::IntegrationRule &ir);

            void postRegistrationSetup(typename Traits::SetupData d,
                                    PHX::FieldManager<Traits> &fm);

            void evaluateFields(typename Traits::EvalData d);

        private:
            typedef typename EvalT::ScalarT ScalarT;

            PHX::MDField<ScalarT, Cell, Point> source;
            int ir_degree, ir_index;
    };

    template <typename EvalT,typename Traits>
    SimpleSource<EvalT,Traits>::
        SimpleSource(const std::string &name,
                     const panzer::IntegrationRule &ir)
    {
        using Teuchos::RCP;

        // 物理场的数据类型
        // scalar标量场
        // vector向量场
        // tensor张量场
        RCP<PHX::DataLayout> data_layout = ir.dl_scalar;
        // 积分阶数
        ir_degree = ir.cubature_degree;

        source = PHX::MDField<ScalarT,Cell,Point>(name,data_layout);
        // 计算完储存到内部中
        this->addEvaluatedField(source);
        
        std::string evalname = "Simple Source";
        this->setName(evalname); 
    }

    template <typename EvalT,typename Traits>
    void SimpleSource<EvalT,Traits>::
            postRegistrationSetup(typename Traits::SetupData sd,           
                                  PHX::FieldManager<Traits>& /* fm */)
    {
        ir_index = panzer::getIntegrationRuleIndex(ir_degree,
                    (*sd.worksets_)[0], this->wda);
    }

    template <typename EvalT,typename Traits>
    void SimpleSource<EvalT,Traits>::
         evaluateFields(typename Traits::EvalData workset)
    {
        using panzer::index_t;
        // 拿到所有积分点的物理坐标
        auto ip_coordinates = workset.int_rules[ir_index]->
                ip_coordinates.get_static_view();
        auto source_v = source.get_static_view();
        
        Kokkos::parallel_for("SimpleSource",workset.num_cells,
        KOKKOS_LAMBDA (const index_t cell)
        {
            for (int point = 0; point < source_v.extent(1); ++ point)
            {
                // x,y方向上的坐标
                const double &x = ip_coordinates(cell,point,0);
                const double &y = ip_coordinates(cell,point,1);

                source_v(cell,point) = 8.0*M_PI*M_PI*std::sin(2.0*M_PI*x)*std::sin(2.0*M_PI*y);
            }
        });
        // 等待所有并行结果计算结束
        Kokkos::fence();
    }
}