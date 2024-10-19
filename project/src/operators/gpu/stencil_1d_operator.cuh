#ifndef STENCIL_OPERATOR_CU_H
#define STENCIL_OPERATOR_CU_H

namespace cudaoperators
{
    void stencil_1d_operator_cu(int *in, int *out, int arraySize, int padding) ;
}  // namespace operator
#endif 
