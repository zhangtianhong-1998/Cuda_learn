#ifndef CMM_OPERATOR_CU_H
#define CMM_OPERATOR_CU_H

namespace cudaoperators
{
    template <typename T>
    void common_matrix_mul_operator_cu(const T *A, const T *B, T *C, int row, int col);

    // 声明模板函数，确保与下面的实例化相匹配
    template <typename T>
    void matrix_row_sum_operator_cu(const T* A, T* sum, const int row);

    template <typename T>
    void matrix_col_sum_operator_cu(const T* A, T* sum, const int col);
}  // namespace operator
#endif 
