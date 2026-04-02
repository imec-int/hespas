module {
  func.func @main(%arg0: tensor<2560x4xbf16>, %arg1: tensor<2560x2432xbf16>) -> tensor<4x1x2432xbf16> {
        %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2560x4xbf16>) -> tensor<4x2560xbf16>
        %1 = stablehlo.dot %0, %arg1, precision = [DEFAULT, DEFAULT] : (tensor<4x2560xbf16>, tensor<2560x2432xbf16>) -> tensor<4x2432xbf16>
        %2 = stablehlo.reshape %1 : (tensor<4x2432xbf16>) -> tensor<4x1x2432xbf16>
        return %2 : tensor<4x1x2432xbf16>
    }
}