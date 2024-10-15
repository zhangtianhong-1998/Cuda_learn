#ifndef CMM_OPERATOR_CU_H
#define CMM_OPERATOR_CU_H

namespace cudaoperators
{
    void common_matrix_mul_operator_cu(const float *A, const float *B, float *C, int row, int col);
}  // namespace operator
#endif 
