#ifndef VEC_ADD_OPERATOR_CU_H
#define VEC_ADD_OPERATOR_CU_H

namespace cudaoperators
{
    template <typename T>
    void vec_add_operator_cu(T* input1, T* input2, T* output, const int size);
    template <typename T>
    void vec_sum_operator_cu(T* input, T* output, const int size);
    template <typename T>
    void vec_max_operator_cu(T* input, T* output, const int size);
}  // namespace operator
#endif 
