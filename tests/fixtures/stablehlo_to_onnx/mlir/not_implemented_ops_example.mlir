module {
  func.func @main() -> (tensor<4xi32>, tensor<i32>, tensor<4xi32>) {
    %iota = stablehlo.iota dim = 0 : tensor<4xi32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %cidx = stablehlo.constant dense<2> : tensor<1xi64>
    %cupd = stablehlo.constant dense<10> : tensor<i32>

    %reduced = stablehlo.reduce(%iota init: %c0) applies stablehlo.add across dimensions = [0] : (tensor<4xi32>, tensor<i32>) -> tensor<i32>
    // Scatter currently falls back in the translator because no lowering is registered.
    %scatter = "stablehlo.scatter"(%iota, %cidx, %cupd) <{
      scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>,
      unique_indices = true
    }> ({
    ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<i32>
      stablehlo.return %sum : tensor<i32>
    }) : (tensor<4xi32>, tensor<1xi64>, tensor<i32>) -> tensor<4xi32>

    return %iota, %reduced, %scatter : tensor<4xi32>, tensor<i32>, tensor<4xi32>
  }
}
