//<class 'PmapFunction'> ['__annotations__', '__call__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__get__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__name__', '__ne__', '__new__', '__qualname__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__signature__', '__sizeof__', '__str__', '__subclasshook__', '__wrapped__', '_cache_clear', '_cache_miss', '_cache_size', '_debug_cache_keys', 'lower', 'trace']
module @pmap_spmd_update attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x784x1024xf32>, %arg1: tensor<1x1024xf32>, %arg2: tensor<1x1024x1024xf32>, %arg3: tensor<1x1024xf32>, %arg4: tensor<1x1024x10xf32>, %arg5: tensor<1x10xf32>, %arg6: tensor<1x128x784xf32>, %arg7: tensor<1x128x10xf32>) -> (tensor<1x784x1024xf32> {jax.result_info = "[0][0]"}, tensor<1x1024xf32> {jax.result_info = "[0][1]"}, tensor<1x1024x1024xf32> {jax.result_info = "[1][0]"}, tensor<1x1024xf32> {jax.result_info = "[1][1]"}, tensor<1x1024x10xf32> {jax.result_info = "[2][0]"}, tensor<1x10xf32> {jax.result_info = "[2][1]"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<1x784x1024xf32>) -> tensor<784x1024xf32>
    %1 = stablehlo.reshape %arg1 : (tensor<1x1024xf32>) -> tensor<1024xf32>
    %2 = stablehlo.reshape %arg2 : (tensor<1x1024x1024xf32>) -> tensor<1024x1024xf32>
    %3 = stablehlo.reshape %arg3 : (tensor<1x1024xf32>) -> tensor<1024xf32>
    %4 = stablehlo.reshape %arg4 : (tensor<1x1024x10xf32>) -> tensor<1024x10xf32>
    %5 = stablehlo.reshape %arg5 : (tensor<1x10xf32>) -> tensor<10xf32>
    %6 = stablehlo.reshape %arg6 : (tensor<1x128x784xf32>) -> tensor<128x784xf32>
    %7 = stablehlo.reshape %arg7 : (tensor<1x128x10xf32>) -> tensor<128x10xf32>
    %8 = stablehlo.dot_general %6, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x784xf32>, tensor<784x1024xf32>) -> tensor<128x1024xf32>
    %9 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<1x1024xf32>) -> tensor<128x1024xf32>
    %11 = stablehlo.add %8, %10 : tensor<128x1024xf32>
    %12 = stablehlo.tanh %11 : tensor<128x1024xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
    %14 = stablehlo.subtract %13, %12 : tensor<128x1024xf32>
    %15 = stablehlo.dot_general %12, %2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x1024xf32>, tensor<1024x1024xf32>) -> tensor<128x1024xf32>
    %16 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<1x1024xf32>) -> tensor<128x1024xf32>
    %18 = stablehlo.add %15, %17 : tensor<128x1024xf32>
    %19 = stablehlo.tanh %18 : tensor<128x1024xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
    %21 = stablehlo.subtract %20, %19 : tensor<128x1024xf32>
    %22 = stablehlo.dot_general %19, %4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x1024xf32>, tensor<1024x10xf32>) -> tensor<128x10xf32>
    %23 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<10xf32>) -> tensor<1x10xf32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0, 1] : (tensor<1x10xf32>) -> tensor<128x10xf32>
    %25 = stablehlo.add %22, %24 : tensor<128x10xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %26 = stablehlo.reduce(%25 init: %cst_1) applies stablehlo.maximum across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %27 = stablehlo.reshape %26 : (tensor<128xf32>) -> tensor<128x1xf32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x10xf32>
    %29 = stablehlo.compare  EQ, %25, %28,  FLOAT : (tensor<128x10xf32>, tensor<128x10xf32>) -> tensor<128x10xi1>
    %30 = stablehlo.convert %29 : (tensor<128x10xi1>) -> tensor<128x10xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %31 = stablehlo.reduce(%30 init: %cst_2) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %32 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %33 = stablehlo.maximum %32, %26 : tensor<128xf32>
    %34 = stablehlo.compare  EQ, %26, %33,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %37 = stablehlo.select %34, %35, %36 : tensor<128xi1>, tensor<128xf32>
    %cst_6 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %39 = stablehlo.compare  EQ, %38, %33,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %cst_7 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %40 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %41 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %42 = stablehlo.select %39, %40, %41 : tensor<128xi1>, tensor<128xf32>
    %43 = stablehlo.divide %37, %42 : tensor<128xf32>
    %44 = stablehlo.broadcast_in_dim %33, dims = [0] : (tensor<128xf32>) -> tensor<128x1xf32>
    %45 = stablehlo.is_finite %44 : (tensor<128x1xf32>) -> tensor<128x1xi1>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %47 = stablehlo.select %45, %44, %46 : tensor<128x1xi1>, tensor<128x1xf32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %48 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %49 = stablehlo.broadcast_in_dim %47, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x10xf32>
    %50 = stablehlo.subtract %25, %49 : tensor<128x10xf32>
    %51 = stablehlo.exponential %50 : tensor<128x10xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %52 = stablehlo.reduce(%51 init: %cst_11) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0] : (tensor<128xf32>) -> tensor<128x1xf32>
    %54 = stablehlo.sign %53 : tensor<128x1xf32>
    %55 = stablehlo.abs %53 : tensor<128x1xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %57 = stablehlo.compare  GE, %53, %56,  FLOAT : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xi1>
    %58 = stablehlo.log %55 : tensor<128x1xf32>
    %59 = stablehlo.add %58, %47 : tensor<128x1xf32>
    %60 = stablehlo.broadcast_in_dim %59, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x10xf32>
    %61 = stablehlo.subtract %25, %60 : tensor<128x10xf32>
    %62 = stablehlo.multiply %61, %7 : tensor<128x10xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %63 = stablehlo.reduce(%62 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %64 = stablehlo.reduce(%63 init: %cst_14) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    %cst_15 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %65 = stablehlo.divide %64, %cst_15 : tensor<f32>
    %66 = stablehlo.negate %65 : tensor<f32>
    %cst_16 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %67 = stablehlo.negate %cst_16 : tensor<f32>
    %cst_17 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %68 = stablehlo.divide %67, %cst_17 : tensor<f32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %70 = stablehlo.broadcast_in_dim %69, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %71 = stablehlo.multiply %70, %7 : tensor<128x10xf32>
    %72 = stablehlo.negate %71 : tensor<128x10xf32>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %73 = stablehlo.reduce(%72 init: %cst_18) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %74 = stablehlo.reshape %73 : (tensor<128xf32>) -> tensor<128x1xf32>
    %75 = stablehlo.divide %74, %55 : tensor<128x1xf32>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %76 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %77 = stablehlo.select %57, %76, %75 : tensor<128x1xi1>, tensor<128x1xf32>
    %78 = stablehlo.select %57, %75, %76 : tensor<128x1xi1>, tensor<128x1xf32>
    %79 = stablehlo.negate %77 : tensor<128x1xf32>
    %80 = stablehlo.add %78, %79 : tensor<128x1xf32>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %81 = stablehlo.reduce(%80 init: %cst_20) applies stablehlo.add across dimensions = [1] : (tensor<128x1xf32>, tensor<f32>) -> tensor<128xf32>
    %82 = stablehlo.broadcast_in_dim %81, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %83 = stablehlo.multiply %82, %51 : tensor<128x10xf32>
    %84 = stablehlo.add %71, %83 : tensor<128x10xf32>
    %cst_21 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %85 = stablehlo.reduce(%84 init: %cst_21) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %86 = stablehlo.reshape %85 : (tensor<10xf32>) -> tensor<1x10xf32>
    %cst_22 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %87 = stablehlo.reduce(%86 init: %cst_22) applies stablehlo.add across dimensions = [0] : (tensor<1x10xf32>, tensor<f32>) -> tensor<10xf32>
    %88 = stablehlo.dot_general %84, %19, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<128x1024xf32>) -> tensor<10x1024xf32>
    %89 = stablehlo.transpose %88, dims = [1, 0] : (tensor<10x1024xf32>) -> tensor<1024x10xf32>
    %90 = stablehlo.dot_general %84, %4, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<1024x10xf32>) -> tensor<128x1024xf32>
    %91 = stablehlo.multiply %90, %21 : tensor<128x1024xf32>
    %92 = stablehlo.multiply %91, %19 : tensor<128x1024xf32>
    %93 = stablehlo.add %91, %92 : tensor<128x1024xf32>
    %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %94 = stablehlo.reduce(%93 init: %cst_23) applies stablehlo.add across dimensions = [0] : (tensor<128x1024xf32>, tensor<f32>) -> tensor<1024xf32>
    %95 = stablehlo.reshape %94 : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %96 = stablehlo.reduce(%95 init: %cst_24) applies stablehlo.add across dimensions = [0] : (tensor<1x1024xf32>, tensor<f32>) -> tensor<1024xf32>
    %97 = stablehlo.dot_general %93, %12, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x1024xf32>, tensor<128x1024xf32>) -> tensor<1024x1024xf32>
    %98 = stablehlo.transpose %97, dims = [1, 0] : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %99 = stablehlo.dot_general %93, %2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1024xf32>, tensor<1024x1024xf32>) -> tensor<128x1024xf32>
    %100 = stablehlo.multiply %99, %14 : tensor<128x1024xf32>
    %101 = stablehlo.multiply %100, %12 : tensor<128x1024xf32>
    %102 = stablehlo.add %100, %101 : tensor<128x1024xf32>
    %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %103 = stablehlo.reduce(%102 init: %cst_25) applies stablehlo.add across dimensions = [0] : (tensor<128x1024xf32>, tensor<f32>) -> tensor<1024xf32>
    %104 = stablehlo.reshape %103 : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %105 = stablehlo.reduce(%104 init: %cst_26) applies stablehlo.add across dimensions = [0] : (tensor<1x1024xf32>, tensor<f32>) -> tensor<1024xf32>
    %106 = stablehlo.dot_general %102, %6, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x1024xf32>, tensor<128x784xf32>) -> tensor<1024x784xf32>
    %107 = stablehlo.transpose %106, dims = [1, 0] : (tensor<1024x784xf32>) -> tensor<784x1024xf32>
    %108 = "stablehlo.all_reduce"(%107) <{replica_groups = dense<0> : tensor<1x1xi64>}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      %138 = stablehlo.add %arg8, %arg9 : tensor<f32>
      stablehlo.return %138 : tensor<f32>
    }) : (tensor<784x1024xf32>) -> tensor<784x1024xf32>
    %109 = "stablehlo.all_reduce"(%105) <{replica_groups = dense<0> : tensor<1x1xi64>}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      %138 = stablehlo.add %arg8, %arg9 : tensor<f32>
      stablehlo.return %138 : tensor<f32>
    }) : (tensor<1024xf32>) -> tensor<1024xf32>
    %110 = "stablehlo.all_reduce"(%98) <{replica_groups = dense<0> : tensor<1x1xi64>}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      %138 = stablehlo.add %arg8, %arg9 : tensor<f32>
      stablehlo.return %138 : tensor<f32>
    }) : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %111 = "stablehlo.all_reduce"(%96) <{replica_groups = dense<0> : tensor<1x1xi64>}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      %138 = stablehlo.add %arg8, %arg9 : tensor<f32>
      stablehlo.return %138 : tensor<f32>
    }) : (tensor<1024xf32>) -> tensor<1024xf32>
    %112 = "stablehlo.all_reduce"(%89) <{replica_groups = dense<0> : tensor<1x1xi64>}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      %138 = stablehlo.add %arg8, %arg9 : tensor<f32>
      stablehlo.return %138 : tensor<f32>
    }) : (tensor<1024x10xf32>) -> tensor<1024x10xf32>
    %113 = "stablehlo.all_reduce"(%87) <{replica_groups = dense<0> : tensor<1x1xi64>}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      %138 = stablehlo.add %arg8, %arg9 : tensor<f32>
      stablehlo.return %138 : tensor<f32>
    }) : (tensor<10xf32>) -> tensor<10xf32>
    %cst_27 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %114 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<784x1024xf32>
    %115 = stablehlo.multiply %114, %108 : tensor<784x1024xf32>
    %116 = stablehlo.subtract %0, %115 : tensor<784x1024xf32>
    %cst_28 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %117 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %118 = stablehlo.multiply %117, %109 : tensor<1024xf32>
    %119 = stablehlo.subtract %1, %118 : tensor<1024xf32>
    %cst_29 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %120 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1024x1024xf32>
    %121 = stablehlo.multiply %120, %110 : tensor<1024x1024xf32>
    %122 = stablehlo.subtract %2, %121 : tensor<1024x1024xf32>
    %cst_30 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %123 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %124 = stablehlo.multiply %123, %111 : tensor<1024xf32>
    %125 = stablehlo.subtract %3, %124 : tensor<1024xf32>
    %cst_31 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %126 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<1024x10xf32>
    %127 = stablehlo.multiply %126, %112 : tensor<1024x10xf32>
    %128 = stablehlo.subtract %4, %127 : tensor<1024x10xf32>
    %cst_32 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %129 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %130 = stablehlo.multiply %129, %113 : tensor<10xf32>
    %131 = stablehlo.subtract %5, %130 : tensor<10xf32>
    %132 = stablehlo.broadcast_in_dim %116, dims = [1, 2] : (tensor<784x1024xf32>) -> tensor<1x784x1024xf32>
    %133 = stablehlo.broadcast_in_dim %119, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %134 = stablehlo.broadcast_in_dim %122, dims = [1, 2] : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %135 = stablehlo.broadcast_in_dim %125, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %136 = stablehlo.broadcast_in_dim %128, dims = [1, 2] : (tensor<1024x10xf32>) -> tensor<1x1024x10xf32>
    %137 = stablehlo.broadcast_in_dim %131, dims = [1] : (tensor<10xf32>) -> tensor<1x10xf32>
    return %132, %133, %134, %135, %136, %137 : tensor<1x784x1024xf32>, tensor<1x1024xf32>, tensor<1x1024x1024xf32>, tensor<1x1024xf32>, tensor<1x1024x10xf32>, tensor<1x10xf32>
  }
}
