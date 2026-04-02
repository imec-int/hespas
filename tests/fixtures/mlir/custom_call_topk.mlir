// Test fixture for stablehlo.custom_call with called_computations attribute
// This tests that private functions referenced by custom_call operations
// are correctly included in minimodules during splitting.

module @test_custom_call_topk {
  // Private comparator function used by TopK custom_call
  func.func private @topk_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
    %0 = stablehlo.compare  GT, %arg0, %arg1,  TOTALORDER : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
    return %0 : tensor<i1>
  }

  func.func @main(%arg0: tensor<2048x8xbf16>) -> (tensor<2048x2xbf16>, tensor<2048x2xi32>) {
    // Custom call to TopK with called_computations referencing the private comparator
    %0 = stablehlo.custom_call @TopK(%arg0) {
      backend_config = "",
      called_computations = [@topk_comparator],
      xla_shape = "(bf16[2048,2]{1,0}, s32[2048,2]{1,0})"
    } : (tensor<2048x8xbf16>) -> tuple<tensor<2048x2xbf16>, tensor<2048x2xi32>>

    // Extract values and indices from the tuple
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<2048x2xbf16>, tensor<2048x2xi32>>) -> tensor<2048x2xbf16>
    %2 = stablehlo.get_tuple_element %0[1] : (tuple<tensor<2048x2xbf16>, tensor<2048x2xi32>>) -> tensor<2048x2xi32>

    return %1, %2 : tensor<2048x2xbf16>, tensor<2048x2xi32>
  }
}
