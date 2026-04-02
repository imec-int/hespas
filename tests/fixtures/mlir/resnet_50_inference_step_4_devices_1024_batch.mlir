module @pmap_inference_step attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 4 : i32} {
  func.func public @main(%arg0: tensor<i32> {mhlo.is_same_data_across_replicas = true}, %arg1: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg2: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg3: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg4: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg5: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg6: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg7: tensor<1x1x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg8: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg9: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg10: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg11: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg12: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg13: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg14: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg15: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg16: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg17: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg18: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg19: tensor<1x1x256x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg20: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg21: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg22: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg23: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg24: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg25: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg26: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg27: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg28: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg29: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg30: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg31: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg32: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg33: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg34: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg35: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg36: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg37: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg38: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg39: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg40: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg41: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg42: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg43: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg44: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg45: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg46: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg47: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg48: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg49: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg50: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg51: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg52: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg53: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg54: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg55: tensor<1x1x1024x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg56: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg57: tensor<1x1x512x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg58: tensor<1x1x1024x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg59: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg60: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg61: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg62: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg63: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg64: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg65: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg66: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg67: tensor<1x1x2048x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg68: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg69: tensor<1x1x512x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg70: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg71: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg72: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg73: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg74: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg75: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg76: tensor<1x1x2048x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg77: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg78: tensor<1x1x512x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg79: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg80: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg81: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg82: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg83: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg84: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg85: tensor<1x1x256x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg86: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg87: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg88: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg89: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg90: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg91: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg92: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg93: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg94: tensor<1x1x256x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg95: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg96: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg97: tensor<1x1x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg98: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg99: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg100: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg101: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg102: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg103: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg104: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg105: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg106: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg107: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg108: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg109: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg110: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg111: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg112: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg113: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg114: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg115: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg116: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg117: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg118: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg119: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg120: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg121: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg122: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg123: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg124: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg125: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg126: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg127: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg128: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg129: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg130: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg131: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg132: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg133: tensor<1x1x512x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg134: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg135: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg136: tensor<1x1x512x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg137: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg138: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg139: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg140: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg141: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg142: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg143: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg144: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg145: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg146: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg147: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg148: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg149: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg150: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg151: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg152: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg153: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg154: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg155: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg156: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg157: tensor<1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg158: tensor<2048x1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg159: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg160: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg161: tensor<7x7x3x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg162: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg163: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg164: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg165: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg166: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg167: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg168: tensor<1x1x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg169: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg170: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg171: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg172: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg173: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg174: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg175: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg176: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg177: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg178: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg179: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg180: tensor<1x1x256x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg181: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg182: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg183: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg184: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg185: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg186: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg187: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg188: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg189: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg190: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg191: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg192: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg193: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg194: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg195: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg196: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg197: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg198: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg199: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg200: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg201: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg202: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg203: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg204: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg205: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg206: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg207: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg208: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg209: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg210: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg211: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg212: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg213: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg214: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg215: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg216: tensor<1x1x1024x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg217: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg218: tensor<1x1x512x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg219: tensor<1x1x1024x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg220: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg221: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg222: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg223: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg224: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg225: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg226: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg227: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg228: tensor<1x1x2048x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg229: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg230: tensor<1x1x512x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg231: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg232: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg233: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg234: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg235: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg236: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg237: tensor<1x1x2048x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg238: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg239: tensor<1x1x512x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg240: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg241: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg242: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg243: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg244: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg245: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg246: tensor<1x1x256x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg247: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg248: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg249: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg250: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg251: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg252: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg253: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg254: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg255: tensor<1x1x256x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg256: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg257: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg258: tensor<1x1x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg259: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg260: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg261: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg262: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg263: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg264: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg265: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg266: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg267: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg268: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg269: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg270: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg271: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg272: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg273: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg274: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg275: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg276: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg277: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg278: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg279: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg280: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg281: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg282: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg283: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg284: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg285: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg286: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg287: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg288: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg289: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg290: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg291: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg292: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg293: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg294: tensor<1x1x512x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg295: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg296: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg297: tensor<1x1x512x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg298: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg299: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg300: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg301: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg302: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg303: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg304: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg305: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg306: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg307: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg308: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg309: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg310: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg311: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg312: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg313: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg314: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg315: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg316: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg317: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg318: tensor<1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg319: tensor<2048x1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg320: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg321: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg322: tensor<7x7x3x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg323: tensor<i32> {mhlo.is_same_data_across_replicas = true}, %arg324: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg325: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg326: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg327: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg328: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg329: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg330: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg331: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg332: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg333: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg334: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg335: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg336: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg337: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg338: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg339: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg340: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg341: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg342: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg343: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg344: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg345: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg346: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg347: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg348: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg349: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg350: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg351: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg352: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg353: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg354: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg355: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg356: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg357: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg358: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg359: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg360: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg361: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg362: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg363: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg364: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg365: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg366: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg367: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg368: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg369: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg370: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg371: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg372: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg373: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg374: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg375: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg376: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg377: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg378: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg379: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg380: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg381: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg382: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg383: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg384: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg385: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg386: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg387: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg388: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg389: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg390: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg391: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg392: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg393: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg394: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg395: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg396: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg397: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg398: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg399: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg400: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg401: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg402: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg403: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg404: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg405: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg406: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg407: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg408: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg409: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg410: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg411: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg412: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg413: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg414: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg415: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg416: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg417: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg418: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg419: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg420: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg421: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg422: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg423: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg424: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg425: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg426: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg427: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg428: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg429: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg430: tensor<1x256x224x224x3xf16>, %arg431: tensor<1x256xi32>) -> (tensor<1x256x1000xf16> {jax.result_info = ""}) {
    %0 = stablehlo.reshape %arg430 : (tensor<1x256x224x224x3xf16>) -> tensor<256x224x224x3xf16>
    %1 = stablehlo.reshape %arg431 : (tensor<1x256xi32>) -> tensor<256xi32>
    %2 = stablehlo.convert %arg161 : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf16>
    %3 = stablehlo.convolution(%0, %2) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x224x224x3xf16>, tensor<7x7x3x64xf16>) -> tensor<256x112x112x64xf16>
    %4 = stablehlo.broadcast_in_dim %arg428, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg429, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %6 = stablehlo.convert %3 : (tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xf32>
    %7 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %8 = stablehlo.subtract %6, %7 : tensor<256x112x112x64xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %10 = stablehlo.add %5, %9 : tensor<1x1x1x64xf32>
    %11 = stablehlo.rsqrt %10 : tensor<1x1x1x64xf32>
    %12 = stablehlo.reshape %arg160 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<1x1x1x64xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %15 = stablehlo.multiply %8, %14 : tensor<256x112x112x64xf32>
    %16 = stablehlo.reshape %arg159 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %18 = stablehlo.add %15, %17 : tensor<256x112x112x64xf32>
    %19 = stablehlo.convert %18 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xf16>
    %20 = call @relu(%19) : (tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xf16>
    %cst_0 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %21 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f16>) -> tensor<f16>
    %22 = "stablehlo.reduce_window"(%20, %21) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
    ^bb0(%arg432: tensor<f16>, %arg433: tensor<f16>):
      %1035 = stablehlo.maximum %arg432, %arg433 : tensor<f16>
      stablehlo.return %1035 : tensor<f16>
    }) : (tensor<256x112x112x64xf16>, tensor<f16>) -> tensor<256x56x56x64xf16>
    %23 = stablehlo.convert %arg7 : (tensor<1x1x64x64xf32>) -> tensor<1x1x64x64xf16>
    %24 = stablehlo.convolution(%22, %23) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x64xf16>) -> tensor<256x56x56x64xf16>
    %25 = stablehlo.broadcast_in_dim %arg324, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %26 = stablehlo.broadcast_in_dim %arg325, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %27 = stablehlo.convert %24 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
    %28 = stablehlo.broadcast_in_dim %25, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %29 = stablehlo.subtract %27, %28 : tensor<256x56x56x64xf32>
    %30 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %31 = stablehlo.add %26, %30 : tensor<1x1x1x64xf32>
    %32 = stablehlo.rsqrt %31 : tensor<1x1x1x64xf32>
    %33 = stablehlo.reshape %arg2 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %34 = stablehlo.multiply %32, %33 : tensor<1x1x1x64xf32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %36 = stablehlo.multiply %29, %35 : tensor<256x56x56x64xf32>
    %37 = stablehlo.reshape %arg1 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %39 = stablehlo.add %36, %38 : tensor<256x56x56x64xf32>
    %40 = stablehlo.convert %39 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
    %41 = call @relu_0(%40) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
    %42 = stablehlo.convert %arg8 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
    %43 = stablehlo.convolution(%41, %42) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
    %44 = stablehlo.broadcast_in_dim %arg326, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %45 = stablehlo.broadcast_in_dim %arg327, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %46 = stablehlo.convert %43 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
    %47 = stablehlo.broadcast_in_dim %44, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %48 = stablehlo.subtract %46, %47 : tensor<256x56x56x64xf32>
    %49 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %50 = stablehlo.add %45, %49 : tensor<1x1x1x64xf32>
    %51 = stablehlo.rsqrt %50 : tensor<1x1x1x64xf32>
    %52 = stablehlo.reshape %arg4 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %53 = stablehlo.multiply %51, %52 : tensor<1x1x1x64xf32>
    %54 = stablehlo.broadcast_in_dim %53, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %55 = stablehlo.multiply %48, %54 : tensor<256x56x56x64xf32>
    %56 = stablehlo.reshape %arg3 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %57 = stablehlo.broadcast_in_dim %56, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %58 = stablehlo.add %55, %57 : tensor<256x56x56x64xf32>
    %59 = stablehlo.convert %58 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
    %60 = call @relu_0(%59) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
    %61 = stablehlo.convert %arg9 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf16>
    %62 = stablehlo.convolution(%60, %61) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x256xf16>) -> tensor<256x56x56x256xf16>
    %63 = stablehlo.broadcast_in_dim %arg328, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %64 = stablehlo.broadcast_in_dim %arg329, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %65 = stablehlo.convert %62 : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf32>
    %66 = stablehlo.broadcast_in_dim %63, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %67 = stablehlo.subtract %65, %66 : tensor<256x56x56x256xf32>
    %68 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %69 = stablehlo.add %64, %68 : tensor<1x1x1x256xf32>
    %70 = stablehlo.rsqrt %69 : tensor<1x1x1x256xf32>
    %71 = stablehlo.reshape %arg6 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %72 = stablehlo.multiply %70, %71 : tensor<1x1x1x256xf32>
    %73 = stablehlo.broadcast_in_dim %72, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %74 = stablehlo.multiply %67, %73 : tensor<256x56x56x256xf32>
    %75 = stablehlo.reshape %arg5 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %76 = stablehlo.broadcast_in_dim %75, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %77 = stablehlo.add %74, %76 : tensor<256x56x56x256xf32>
    %78 = stablehlo.convert %77 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xf16>
    %79 = stablehlo.convert %arg10 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf16>
    %80 = stablehlo.convolution(%22, %79) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x256xf16>) -> tensor<256x56x56x256xf16>
    %81 = stablehlo.broadcast_in_dim %arg330, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %82 = stablehlo.broadcast_in_dim %arg331, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %83 = stablehlo.convert %80 : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf32>
    %84 = stablehlo.broadcast_in_dim %81, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %85 = stablehlo.subtract %83, %84 : tensor<256x56x56x256xf32>
    %86 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %87 = stablehlo.add %82, %86 : tensor<1x1x1x256xf32>
    %88 = stablehlo.rsqrt %87 : tensor<1x1x1x256xf32>
    %89 = stablehlo.reshape %arg12 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %90 = stablehlo.multiply %88, %89 : tensor<1x1x1x256xf32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %92 = stablehlo.multiply %85, %91 : tensor<256x56x56x256xf32>
    %93 = stablehlo.reshape %arg11 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %94 = stablehlo.broadcast_in_dim %93, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %95 = stablehlo.add %92, %94 : tensor<256x56x56x256xf32>
    %96 = stablehlo.convert %95 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xf16>
    %97 = stablehlo.add %96, %78 : tensor<256x56x56x256xf16>
    %98 = call @relu_1(%97) : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf16>
    %99 = stablehlo.convert %arg19 : (tensor<1x1x256x64xf32>) -> tensor<1x1x256x64xf16>
    %100 = stablehlo.convolution(%98, %99) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xf16>, tensor<1x1x256x64xf16>) -> tensor<256x56x56x64xf16>
    %101 = stablehlo.broadcast_in_dim %arg332, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %102 = stablehlo.broadcast_in_dim %arg333, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %103 = stablehlo.convert %100 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
    %104 = stablehlo.broadcast_in_dim %101, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %105 = stablehlo.subtract %103, %104 : tensor<256x56x56x64xf32>
    %106 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %107 = stablehlo.add %102, %106 : tensor<1x1x1x64xf32>
    %108 = stablehlo.rsqrt %107 : tensor<1x1x1x64xf32>
    %109 = stablehlo.reshape %arg14 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %110 = stablehlo.multiply %108, %109 : tensor<1x1x1x64xf32>
    %111 = stablehlo.broadcast_in_dim %110, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %112 = stablehlo.multiply %105, %111 : tensor<256x56x56x64xf32>
    %113 = stablehlo.reshape %arg13 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %114 = stablehlo.broadcast_in_dim %113, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %115 = stablehlo.add %112, %114 : tensor<256x56x56x64xf32>
    %116 = stablehlo.convert %115 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
    %117 = call @relu_0(%116) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
    %118 = stablehlo.convert %arg20 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
    %119 = stablehlo.convolution(%117, %118) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
    %120 = stablehlo.broadcast_in_dim %arg334, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %121 = stablehlo.broadcast_in_dim %arg335, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %122 = stablehlo.convert %119 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
    %123 = stablehlo.broadcast_in_dim %120, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %124 = stablehlo.subtract %122, %123 : tensor<256x56x56x64xf32>
    %125 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %126 = stablehlo.add %121, %125 : tensor<1x1x1x64xf32>
    %127 = stablehlo.rsqrt %126 : tensor<1x1x1x64xf32>
    %128 = stablehlo.reshape %arg16 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %129 = stablehlo.multiply %127, %128 : tensor<1x1x1x64xf32>
    %130 = stablehlo.broadcast_in_dim %129, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %131 = stablehlo.multiply %124, %130 : tensor<256x56x56x64xf32>
    %132 = stablehlo.reshape %arg15 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %133 = stablehlo.broadcast_in_dim %132, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %134 = stablehlo.add %131, %133 : tensor<256x56x56x64xf32>
    %135 = stablehlo.convert %134 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
    %136 = call @relu_0(%135) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
    %137 = stablehlo.convert %arg21 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf16>
    %138 = stablehlo.convolution(%136, %137) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x256xf16>) -> tensor<256x56x56x256xf16>
    %139 = stablehlo.broadcast_in_dim %arg336, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %140 = stablehlo.broadcast_in_dim %arg337, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %141 = stablehlo.convert %138 : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf32>
    %142 = stablehlo.broadcast_in_dim %139, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %143 = stablehlo.subtract %141, %142 : tensor<256x56x56x256xf32>
    %144 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %145 = stablehlo.add %140, %144 : tensor<1x1x1x256xf32>
    %146 = stablehlo.rsqrt %145 : tensor<1x1x1x256xf32>
    %147 = stablehlo.reshape %arg18 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %148 = stablehlo.multiply %146, %147 : tensor<1x1x1x256xf32>
    %149 = stablehlo.broadcast_in_dim %148, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %150 = stablehlo.multiply %143, %149 : tensor<256x56x56x256xf32>
    %151 = stablehlo.reshape %arg17 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %152 = stablehlo.broadcast_in_dim %151, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %153 = stablehlo.add %150, %152 : tensor<256x56x56x256xf32>
    %154 = stablehlo.convert %153 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xf16>
    %155 = stablehlo.add %98, %154 : tensor<256x56x56x256xf16>
    %156 = call @relu_1(%155) : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf16>
    %157 = stablehlo.convert %arg85 : (tensor<1x1x256x64xf32>) -> tensor<1x1x256x64xf16>
    %158 = stablehlo.convolution(%156, %157) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xf16>, tensor<1x1x256x64xf16>) -> tensor<256x56x56x64xf16>
    %159 = stablehlo.broadcast_in_dim %arg376, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %160 = stablehlo.broadcast_in_dim %arg377, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %161 = stablehlo.convert %158 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
    %162 = stablehlo.broadcast_in_dim %159, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %163 = stablehlo.subtract %161, %162 : tensor<256x56x56x64xf32>
    %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %165 = stablehlo.add %160, %164 : tensor<1x1x1x64xf32>
    %166 = stablehlo.rsqrt %165 : tensor<1x1x1x64xf32>
    %167 = stablehlo.reshape %arg80 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %168 = stablehlo.multiply %166, %167 : tensor<1x1x1x64xf32>
    %169 = stablehlo.broadcast_in_dim %168, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %170 = stablehlo.multiply %163, %169 : tensor<256x56x56x64xf32>
    %171 = stablehlo.reshape %arg79 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %172 = stablehlo.broadcast_in_dim %171, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %173 = stablehlo.add %170, %172 : tensor<256x56x56x64xf32>
    %174 = stablehlo.convert %173 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
    %175 = call @relu_0(%174) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
    %176 = stablehlo.convert %arg86 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
    %177 = stablehlo.convolution(%175, %176) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
    %178 = stablehlo.broadcast_in_dim %arg378, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %179 = stablehlo.broadcast_in_dim %arg379, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %180 = stablehlo.convert %177 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
    %181 = stablehlo.broadcast_in_dim %178, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %182 = stablehlo.subtract %180, %181 : tensor<256x56x56x64xf32>
    %183 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %184 = stablehlo.add %179, %183 : tensor<1x1x1x64xf32>
    %185 = stablehlo.rsqrt %184 : tensor<1x1x1x64xf32>
    %186 = stablehlo.reshape %arg82 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %187 = stablehlo.multiply %185, %186 : tensor<1x1x1x64xf32>
    %188 = stablehlo.broadcast_in_dim %187, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %189 = stablehlo.multiply %182, %188 : tensor<256x56x56x64xf32>
    %190 = stablehlo.reshape %arg81 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %191 = stablehlo.broadcast_in_dim %190, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %192 = stablehlo.add %189, %191 : tensor<256x56x56x64xf32>
    %193 = stablehlo.convert %192 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
    %194 = call @relu_0(%193) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
    %195 = stablehlo.convert %arg87 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf16>
    %196 = stablehlo.convolution(%194, %195) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x256xf16>) -> tensor<256x56x56x256xf16>
    %197 = stablehlo.broadcast_in_dim %arg380, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %198 = stablehlo.broadcast_in_dim %arg381, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %199 = stablehlo.convert %196 : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf32>
    %200 = stablehlo.broadcast_in_dim %197, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %201 = stablehlo.subtract %199, %200 : tensor<256x56x56x256xf32>
    %202 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %203 = stablehlo.add %198, %202 : tensor<1x1x1x256xf32>
    %204 = stablehlo.rsqrt %203 : tensor<1x1x1x256xf32>
    %205 = stablehlo.reshape %arg84 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %206 = stablehlo.multiply %204, %205 : tensor<1x1x1x256xf32>
    %207 = stablehlo.broadcast_in_dim %206, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %208 = stablehlo.multiply %201, %207 : tensor<256x56x56x256xf32>
    %209 = stablehlo.reshape %arg83 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %210 = stablehlo.broadcast_in_dim %209, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %211 = stablehlo.add %208, %210 : tensor<256x56x56x256xf32>
    %212 = stablehlo.convert %211 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xf16>
    %213 = stablehlo.add %156, %212 : tensor<256x56x56x256xf16>
    %214 = call @relu_1(%213) : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf16>
    %215 = stablehlo.convert %arg94 : (tensor<1x1x256x128xf32>) -> tensor<1x1x256x128xf16>
    %216 = stablehlo.convolution(%214, %215) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xf16>, tensor<1x1x256x128xf16>) -> tensor<256x56x56x128xf16>
    %217 = stablehlo.broadcast_in_dim %arg382, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %218 = stablehlo.broadcast_in_dim %arg383, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %219 = stablehlo.convert %216 : (tensor<256x56x56x128xf16>) -> tensor<256x56x56x128xf32>
    %220 = stablehlo.broadcast_in_dim %217, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x56x56x128xf32>
    %221 = stablehlo.subtract %219, %220 : tensor<256x56x56x128xf32>
    %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %223 = stablehlo.add %218, %222 : tensor<1x1x1x128xf32>
    %224 = stablehlo.rsqrt %223 : tensor<1x1x1x128xf32>
    %225 = stablehlo.reshape %arg89 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %226 = stablehlo.multiply %224, %225 : tensor<1x1x1x128xf32>
    %227 = stablehlo.broadcast_in_dim %226, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x56x56x128xf32>
    %228 = stablehlo.multiply %221, %227 : tensor<256x56x56x128xf32>
    %229 = stablehlo.reshape %arg88 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %230 = stablehlo.broadcast_in_dim %229, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x56x56x128xf32>
    %231 = stablehlo.add %228, %230 : tensor<256x56x56x128xf32>
    %232 = stablehlo.convert %231 : (tensor<256x56x56x128xf32>) -> tensor<256x56x56x128xf16>
    %233 = call @relu_2(%232) : (tensor<256x56x56x128xf16>) -> tensor<256x56x56x128xf16>
    %234 = stablehlo.convert %arg95 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
    %235 = stablehlo.convolution(%233, %234) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
    %236 = stablehlo.broadcast_in_dim %arg384, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %237 = stablehlo.broadcast_in_dim %arg385, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %238 = stablehlo.convert %235 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
    %239 = stablehlo.broadcast_in_dim %236, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %240 = stablehlo.subtract %238, %239 : tensor<256x28x28x128xf32>
    %241 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %242 = stablehlo.add %237, %241 : tensor<1x1x1x128xf32>
    %243 = stablehlo.rsqrt %242 : tensor<1x1x1x128xf32>
    %244 = stablehlo.reshape %arg91 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %245 = stablehlo.multiply %243, %244 : tensor<1x1x1x128xf32>
    %246 = stablehlo.broadcast_in_dim %245, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %247 = stablehlo.multiply %240, %246 : tensor<256x28x28x128xf32>
    %248 = stablehlo.reshape %arg90 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %249 = stablehlo.broadcast_in_dim %248, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %250 = stablehlo.add %247, %249 : tensor<256x28x28x128xf32>
    %251 = stablehlo.convert %250 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
    %252 = call @relu_3(%251) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
    %253 = stablehlo.convert %arg96 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
    %254 = stablehlo.convolution(%252, %253) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
    %255 = stablehlo.broadcast_in_dim %arg386, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %256 = stablehlo.broadcast_in_dim %arg387, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %257 = stablehlo.convert %254 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
    %258 = stablehlo.broadcast_in_dim %255, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %259 = stablehlo.subtract %257, %258 : tensor<256x28x28x512xf32>
    %260 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %261 = stablehlo.add %256, %260 : tensor<1x1x1x512xf32>
    %262 = stablehlo.rsqrt %261 : tensor<1x1x1x512xf32>
    %263 = stablehlo.reshape %arg93 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %264 = stablehlo.multiply %262, %263 : tensor<1x1x1x512xf32>
    %265 = stablehlo.broadcast_in_dim %264, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %266 = stablehlo.multiply %259, %265 : tensor<256x28x28x512xf32>
    %267 = stablehlo.reshape %arg92 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %268 = stablehlo.broadcast_in_dim %267, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %269 = stablehlo.add %266, %268 : tensor<256x28x28x512xf32>
    %270 = stablehlo.convert %269 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
    %271 = stablehlo.convert %arg97 : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf16>
    %272 = stablehlo.convolution(%214, %271) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xf16>, tensor<1x1x256x512xf16>) -> tensor<256x28x28x512xf16>
    %273 = stablehlo.broadcast_in_dim %arg388, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %274 = stablehlo.broadcast_in_dim %arg389, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %275 = stablehlo.convert %272 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
    %276 = stablehlo.broadcast_in_dim %273, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %277 = stablehlo.subtract %275, %276 : tensor<256x28x28x512xf32>
    %278 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %279 = stablehlo.add %274, %278 : tensor<1x1x1x512xf32>
    %280 = stablehlo.rsqrt %279 : tensor<1x1x1x512xf32>
    %281 = stablehlo.reshape %arg99 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %282 = stablehlo.multiply %280, %281 : tensor<1x1x1x512xf32>
    %283 = stablehlo.broadcast_in_dim %282, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %284 = stablehlo.multiply %277, %283 : tensor<256x28x28x512xf32>
    %285 = stablehlo.reshape %arg98 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %287 = stablehlo.add %284, %286 : tensor<256x28x28x512xf32>
    %288 = stablehlo.convert %287 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
    %289 = stablehlo.add %288, %270 : tensor<256x28x28x512xf16>
    %290 = call @relu_4(%289) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
    %291 = stablehlo.convert %arg106 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
    %292 = stablehlo.convolution(%290, %291) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
    %293 = stablehlo.broadcast_in_dim %arg390, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %294 = stablehlo.broadcast_in_dim %arg391, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %295 = stablehlo.convert %292 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
    %296 = stablehlo.broadcast_in_dim %293, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %297 = stablehlo.subtract %295, %296 : tensor<256x28x28x128xf32>
    %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %299 = stablehlo.add %294, %298 : tensor<1x1x1x128xf32>
    %300 = stablehlo.rsqrt %299 : tensor<1x1x1x128xf32>
    %301 = stablehlo.reshape %arg101 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %302 = stablehlo.multiply %300, %301 : tensor<1x1x1x128xf32>
    %303 = stablehlo.broadcast_in_dim %302, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %304 = stablehlo.multiply %297, %303 : tensor<256x28x28x128xf32>
    %305 = stablehlo.reshape %arg100 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %307 = stablehlo.add %304, %306 : tensor<256x28x28x128xf32>
    %308 = stablehlo.convert %307 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
    %309 = call @relu_3(%308) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
    %310 = stablehlo.convert %arg107 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
    %311 = stablehlo.convolution(%309, %310) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
    %312 = stablehlo.broadcast_in_dim %arg392, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %313 = stablehlo.broadcast_in_dim %arg393, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %314 = stablehlo.convert %311 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
    %315 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %316 = stablehlo.subtract %314, %315 : tensor<256x28x28x128xf32>
    %317 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %318 = stablehlo.add %313, %317 : tensor<1x1x1x128xf32>
    %319 = stablehlo.rsqrt %318 : tensor<1x1x1x128xf32>
    %320 = stablehlo.reshape %arg103 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %321 = stablehlo.multiply %319, %320 : tensor<1x1x1x128xf32>
    %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %323 = stablehlo.multiply %316, %322 : tensor<256x28x28x128xf32>
    %324 = stablehlo.reshape %arg102 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %325 = stablehlo.broadcast_in_dim %324, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %326 = stablehlo.add %323, %325 : tensor<256x28x28x128xf32>
    %327 = stablehlo.convert %326 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
    %328 = call @relu_3(%327) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
    %329 = stablehlo.convert %arg108 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
    %330 = stablehlo.convolution(%328, %329) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
    %331 = stablehlo.broadcast_in_dim %arg394, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %332 = stablehlo.broadcast_in_dim %arg395, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %333 = stablehlo.convert %330 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
    %334 = stablehlo.broadcast_in_dim %331, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %335 = stablehlo.subtract %333, %334 : tensor<256x28x28x512xf32>
    %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %337 = stablehlo.add %332, %336 : tensor<1x1x1x512xf32>
    %338 = stablehlo.rsqrt %337 : tensor<1x1x1x512xf32>
    %339 = stablehlo.reshape %arg105 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %340 = stablehlo.multiply %338, %339 : tensor<1x1x1x512xf32>
    %341 = stablehlo.broadcast_in_dim %340, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %342 = stablehlo.multiply %335, %341 : tensor<256x28x28x512xf32>
    %343 = stablehlo.reshape %arg104 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %344 = stablehlo.broadcast_in_dim %343, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %345 = stablehlo.add %342, %344 : tensor<256x28x28x512xf32>
    %346 = stablehlo.convert %345 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
    %347 = stablehlo.add %290, %346 : tensor<256x28x28x512xf16>
    %348 = call @relu_4(%347) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
    %349 = stablehlo.convert %arg115 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
    %350 = stablehlo.convolution(%348, %349) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
    %351 = stablehlo.broadcast_in_dim %arg396, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %352 = stablehlo.broadcast_in_dim %arg397, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %353 = stablehlo.convert %350 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
    %354 = stablehlo.broadcast_in_dim %351, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %355 = stablehlo.subtract %353, %354 : tensor<256x28x28x128xf32>
    %356 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %357 = stablehlo.add %352, %356 : tensor<1x1x1x128xf32>
    %358 = stablehlo.rsqrt %357 : tensor<1x1x1x128xf32>
    %359 = stablehlo.reshape %arg110 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %360 = stablehlo.multiply %358, %359 : tensor<1x1x1x128xf32>
    %361 = stablehlo.broadcast_in_dim %360, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %362 = stablehlo.multiply %355, %361 : tensor<256x28x28x128xf32>
    %363 = stablehlo.reshape %arg109 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %364 = stablehlo.broadcast_in_dim %363, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %365 = stablehlo.add %362, %364 : tensor<256x28x28x128xf32>
    %366 = stablehlo.convert %365 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
    %367 = call @relu_3(%366) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
    %368 = stablehlo.convert %arg116 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
    %369 = stablehlo.convolution(%367, %368) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
    %370 = stablehlo.broadcast_in_dim %arg398, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %371 = stablehlo.broadcast_in_dim %arg399, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %372 = stablehlo.convert %369 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
    %373 = stablehlo.broadcast_in_dim %370, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %374 = stablehlo.subtract %372, %373 : tensor<256x28x28x128xf32>
    %375 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %376 = stablehlo.add %371, %375 : tensor<1x1x1x128xf32>
    %377 = stablehlo.rsqrt %376 : tensor<1x1x1x128xf32>
    %378 = stablehlo.reshape %arg112 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %379 = stablehlo.multiply %377, %378 : tensor<1x1x1x128xf32>
    %380 = stablehlo.broadcast_in_dim %379, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %381 = stablehlo.multiply %374, %380 : tensor<256x28x28x128xf32>
    %382 = stablehlo.reshape %arg111 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %383 = stablehlo.broadcast_in_dim %382, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %384 = stablehlo.add %381, %383 : tensor<256x28x28x128xf32>
    %385 = stablehlo.convert %384 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
    %386 = call @relu_3(%385) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
    %387 = stablehlo.convert %arg117 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
    %388 = stablehlo.convolution(%386, %387) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
    %389 = stablehlo.broadcast_in_dim %arg400, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %390 = stablehlo.broadcast_in_dim %arg401, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %391 = stablehlo.convert %388 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
    %392 = stablehlo.broadcast_in_dim %389, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %393 = stablehlo.subtract %391, %392 : tensor<256x28x28x512xf32>
    %394 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %395 = stablehlo.add %390, %394 : tensor<1x1x1x512xf32>
    %396 = stablehlo.rsqrt %395 : tensor<1x1x1x512xf32>
    %397 = stablehlo.reshape %arg114 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %398 = stablehlo.multiply %396, %397 : tensor<1x1x1x512xf32>
    %399 = stablehlo.broadcast_in_dim %398, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %400 = stablehlo.multiply %393, %399 : tensor<256x28x28x512xf32>
    %401 = stablehlo.reshape %arg113 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %403 = stablehlo.add %400, %402 : tensor<256x28x28x512xf32>
    %404 = stablehlo.convert %403 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
    %405 = stablehlo.add %348, %404 : tensor<256x28x28x512xf16>
    %406 = call @relu_4(%405) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
    %407 = stablehlo.convert %arg124 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
    %408 = stablehlo.convolution(%406, %407) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
    %409 = stablehlo.broadcast_in_dim %arg402, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %410 = stablehlo.broadcast_in_dim %arg403, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %411 = stablehlo.convert %408 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
    %412 = stablehlo.broadcast_in_dim %409, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %413 = stablehlo.subtract %411, %412 : tensor<256x28x28x128xf32>
    %414 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %415 = stablehlo.add %410, %414 : tensor<1x1x1x128xf32>
    %416 = stablehlo.rsqrt %415 : tensor<1x1x1x128xf32>
    %417 = stablehlo.reshape %arg119 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %418 = stablehlo.multiply %416, %417 : tensor<1x1x1x128xf32>
    %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %420 = stablehlo.multiply %413, %419 : tensor<256x28x28x128xf32>
    %421 = stablehlo.reshape %arg118 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %423 = stablehlo.add %420, %422 : tensor<256x28x28x128xf32>
    %424 = stablehlo.convert %423 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
    %425 = call @relu_3(%424) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
    %426 = stablehlo.convert %arg125 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
    %427 = stablehlo.convolution(%425, %426) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
    %428 = stablehlo.broadcast_in_dim %arg404, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %429 = stablehlo.broadcast_in_dim %arg405, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %430 = stablehlo.convert %427 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
    %431 = stablehlo.broadcast_in_dim %428, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %432 = stablehlo.subtract %430, %431 : tensor<256x28x28x128xf32>
    %433 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %434 = stablehlo.add %429, %433 : tensor<1x1x1x128xf32>
    %435 = stablehlo.rsqrt %434 : tensor<1x1x1x128xf32>
    %436 = stablehlo.reshape %arg121 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %437 = stablehlo.multiply %435, %436 : tensor<1x1x1x128xf32>
    %438 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %439 = stablehlo.multiply %432, %438 : tensor<256x28x28x128xf32>
    %440 = stablehlo.reshape %arg120 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %441 = stablehlo.broadcast_in_dim %440, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %442 = stablehlo.add %439, %441 : tensor<256x28x28x128xf32>
    %443 = stablehlo.convert %442 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
    %444 = call @relu_3(%443) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
    %445 = stablehlo.convert %arg126 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
    %446 = stablehlo.convolution(%444, %445) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
    %447 = stablehlo.broadcast_in_dim %arg406, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %448 = stablehlo.broadcast_in_dim %arg407, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %449 = stablehlo.convert %446 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
    %450 = stablehlo.broadcast_in_dim %447, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %451 = stablehlo.subtract %449, %450 : tensor<256x28x28x512xf32>
    %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %453 = stablehlo.add %448, %452 : tensor<1x1x1x512xf32>
    %454 = stablehlo.rsqrt %453 : tensor<1x1x1x512xf32>
    %455 = stablehlo.reshape %arg123 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %456 = stablehlo.multiply %454, %455 : tensor<1x1x1x512xf32>
    %457 = stablehlo.broadcast_in_dim %456, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %458 = stablehlo.multiply %451, %457 : tensor<256x28x28x512xf32>
    %459 = stablehlo.reshape %arg122 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %460 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %461 = stablehlo.add %458, %460 : tensor<256x28x28x512xf32>
    %462 = stablehlo.convert %461 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
    %463 = stablehlo.add %406, %462 : tensor<256x28x28x512xf16>
    %464 = call @relu_4(%463) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
    %465 = stablehlo.convert %arg133 : (tensor<1x1x512x256xf32>) -> tensor<1x1x512x256xf16>
    %466 = stablehlo.convolution(%464, %465) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x256xf16>) -> tensor<256x28x28x256xf16>
    %467 = stablehlo.broadcast_in_dim %arg408, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %468 = stablehlo.broadcast_in_dim %arg409, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %469 = stablehlo.convert %466 : (tensor<256x28x28x256xf16>) -> tensor<256x28x28x256xf32>
    %470 = stablehlo.broadcast_in_dim %467, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x28x28x256xf32>
    %471 = stablehlo.subtract %469, %470 : tensor<256x28x28x256xf32>
    %472 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %473 = stablehlo.add %468, %472 : tensor<1x1x1x256xf32>
    %474 = stablehlo.rsqrt %473 : tensor<1x1x1x256xf32>
    %475 = stablehlo.reshape %arg128 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %476 = stablehlo.multiply %474, %475 : tensor<1x1x1x256xf32>
    %477 = stablehlo.broadcast_in_dim %476, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x28x28x256xf32>
    %478 = stablehlo.multiply %471, %477 : tensor<256x28x28x256xf32>
    %479 = stablehlo.reshape %arg127 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %480 = stablehlo.broadcast_in_dim %479, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x28x28x256xf32>
    %481 = stablehlo.add %478, %480 : tensor<256x28x28x256xf32>
    %482 = stablehlo.convert %481 : (tensor<256x28x28x256xf32>) -> tensor<256x28x28x256xf16>
    %483 = call @relu_5(%482) : (tensor<256x28x28x256xf16>) -> tensor<256x28x28x256xf16>
    %484 = stablehlo.convert %arg134 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
    %485 = stablehlo.convolution(%483, %484) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
    %486 = stablehlo.broadcast_in_dim %arg410, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %487 = stablehlo.broadcast_in_dim %arg411, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %488 = stablehlo.convert %485 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
    %489 = stablehlo.broadcast_in_dim %486, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %490 = stablehlo.subtract %488, %489 : tensor<256x14x14x256xf32>
    %491 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %492 = stablehlo.add %487, %491 : tensor<1x1x1x256xf32>
    %493 = stablehlo.rsqrt %492 : tensor<1x1x1x256xf32>
    %494 = stablehlo.reshape %arg130 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %495 = stablehlo.multiply %493, %494 : tensor<1x1x1x256xf32>
    %496 = stablehlo.broadcast_in_dim %495, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %497 = stablehlo.multiply %490, %496 : tensor<256x14x14x256xf32>
    %498 = stablehlo.reshape %arg129 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %499 = stablehlo.broadcast_in_dim %498, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %500 = stablehlo.add %497, %499 : tensor<256x14x14x256xf32>
    %501 = stablehlo.convert %500 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
    %502 = call @relu_6(%501) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
    %503 = stablehlo.convert %arg135 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
    %504 = stablehlo.convolution(%502, %503) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
    %505 = stablehlo.broadcast_in_dim %arg412, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %506 = stablehlo.broadcast_in_dim %arg413, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %507 = stablehlo.convert %504 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
    %508 = stablehlo.broadcast_in_dim %505, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %509 = stablehlo.subtract %507, %508 : tensor<256x14x14x1024xf32>
    %510 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %511 = stablehlo.add %506, %510 : tensor<1x1x1x1024xf32>
    %512 = stablehlo.rsqrt %511 : tensor<1x1x1x1024xf32>
    %513 = stablehlo.reshape %arg132 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %514 = stablehlo.multiply %512, %513 : tensor<1x1x1x1024xf32>
    %515 = stablehlo.broadcast_in_dim %514, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %516 = stablehlo.multiply %509, %515 : tensor<256x14x14x1024xf32>
    %517 = stablehlo.reshape %arg131 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %518 = stablehlo.broadcast_in_dim %517, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %519 = stablehlo.add %516, %518 : tensor<256x14x14x1024xf32>
    %520 = stablehlo.convert %519 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
    %521 = stablehlo.convert %arg136 : (tensor<1x1x512x1024xf32>) -> tensor<1x1x512x1024xf16>
    %522 = stablehlo.convolution(%464, %521) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x1024xf16>) -> tensor<256x14x14x1024xf16>
    %523 = stablehlo.broadcast_in_dim %arg414, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %524 = stablehlo.broadcast_in_dim %arg415, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %525 = stablehlo.convert %522 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
    %526 = stablehlo.broadcast_in_dim %523, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %527 = stablehlo.subtract %525, %526 : tensor<256x14x14x1024xf32>
    %528 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %529 = stablehlo.add %524, %528 : tensor<1x1x1x1024xf32>
    %530 = stablehlo.rsqrt %529 : tensor<1x1x1x1024xf32>
    %531 = stablehlo.reshape %arg138 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %532 = stablehlo.multiply %530, %531 : tensor<1x1x1x1024xf32>
    %533 = stablehlo.broadcast_in_dim %532, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %534 = stablehlo.multiply %527, %533 : tensor<256x14x14x1024xf32>
    %535 = stablehlo.reshape %arg137 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %536 = stablehlo.broadcast_in_dim %535, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %537 = stablehlo.add %534, %536 : tensor<256x14x14x1024xf32>
    %538 = stablehlo.convert %537 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
    %539 = stablehlo.add %538, %520 : tensor<256x14x14x1024xf16>
    %540 = call @relu_7(%539) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
    %541 = stablehlo.convert %arg145 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
    %542 = stablehlo.convolution(%540, %541) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
    %543 = stablehlo.broadcast_in_dim %arg416, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %544 = stablehlo.broadcast_in_dim %arg417, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %545 = stablehlo.convert %542 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
    %546 = stablehlo.broadcast_in_dim %543, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %547 = stablehlo.subtract %545, %546 : tensor<256x14x14x256xf32>
    %548 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %549 = stablehlo.add %544, %548 : tensor<1x1x1x256xf32>
    %550 = stablehlo.rsqrt %549 : tensor<1x1x1x256xf32>
    %551 = stablehlo.reshape %arg140 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %552 = stablehlo.multiply %550, %551 : tensor<1x1x1x256xf32>
    %553 = stablehlo.broadcast_in_dim %552, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %554 = stablehlo.multiply %547, %553 : tensor<256x14x14x256xf32>
    %555 = stablehlo.reshape %arg139 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %556 = stablehlo.broadcast_in_dim %555, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %557 = stablehlo.add %554, %556 : tensor<256x14x14x256xf32>
    %558 = stablehlo.convert %557 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
    %559 = call @relu_6(%558) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
    %560 = stablehlo.convert %arg146 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
    %561 = stablehlo.convolution(%559, %560) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
    %562 = stablehlo.broadcast_in_dim %arg418, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %563 = stablehlo.broadcast_in_dim %arg419, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %564 = stablehlo.convert %561 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
    %565 = stablehlo.broadcast_in_dim %562, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %566 = stablehlo.subtract %564, %565 : tensor<256x14x14x256xf32>
    %567 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %568 = stablehlo.add %563, %567 : tensor<1x1x1x256xf32>
    %569 = stablehlo.rsqrt %568 : tensor<1x1x1x256xf32>
    %570 = stablehlo.reshape %arg142 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %571 = stablehlo.multiply %569, %570 : tensor<1x1x1x256xf32>
    %572 = stablehlo.broadcast_in_dim %571, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %573 = stablehlo.multiply %566, %572 : tensor<256x14x14x256xf32>
    %574 = stablehlo.reshape %arg141 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %575 = stablehlo.broadcast_in_dim %574, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %576 = stablehlo.add %573, %575 : tensor<256x14x14x256xf32>
    %577 = stablehlo.convert %576 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
    %578 = call @relu_6(%577) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
    %579 = stablehlo.convert %arg147 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
    %580 = stablehlo.convolution(%578, %579) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
    %581 = stablehlo.broadcast_in_dim %arg420, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %582 = stablehlo.broadcast_in_dim %arg421, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %583 = stablehlo.convert %580 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
    %584 = stablehlo.broadcast_in_dim %581, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %585 = stablehlo.subtract %583, %584 : tensor<256x14x14x1024xf32>
    %586 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %587 = stablehlo.add %582, %586 : tensor<1x1x1x1024xf32>
    %588 = stablehlo.rsqrt %587 : tensor<1x1x1x1024xf32>
    %589 = stablehlo.reshape %arg144 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %590 = stablehlo.multiply %588, %589 : tensor<1x1x1x1024xf32>
    %591 = stablehlo.broadcast_in_dim %590, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %592 = stablehlo.multiply %585, %591 : tensor<256x14x14x1024xf32>
    %593 = stablehlo.reshape %arg143 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %594 = stablehlo.broadcast_in_dim %593, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %595 = stablehlo.add %592, %594 : tensor<256x14x14x1024xf32>
    %596 = stablehlo.convert %595 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
    %597 = stablehlo.add %540, %596 : tensor<256x14x14x1024xf16>
    %598 = call @relu_7(%597) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
    %599 = stablehlo.convert %arg154 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
    %600 = stablehlo.convolution(%598, %599) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
    %601 = stablehlo.broadcast_in_dim %arg422, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %602 = stablehlo.broadcast_in_dim %arg423, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %603 = stablehlo.convert %600 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
    %604 = stablehlo.broadcast_in_dim %601, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %605 = stablehlo.subtract %603, %604 : tensor<256x14x14x256xf32>
    %606 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %607 = stablehlo.add %602, %606 : tensor<1x1x1x256xf32>
    %608 = stablehlo.rsqrt %607 : tensor<1x1x1x256xf32>
    %609 = stablehlo.reshape %arg149 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %610 = stablehlo.multiply %608, %609 : tensor<1x1x1x256xf32>
    %611 = stablehlo.broadcast_in_dim %610, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %612 = stablehlo.multiply %605, %611 : tensor<256x14x14x256xf32>
    %613 = stablehlo.reshape %arg148 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %614 = stablehlo.broadcast_in_dim %613, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %615 = stablehlo.add %612, %614 : tensor<256x14x14x256xf32>
    %616 = stablehlo.convert %615 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
    %617 = call @relu_6(%616) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
    %618 = stablehlo.convert %arg155 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
    %619 = stablehlo.convolution(%617, %618) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
    %620 = stablehlo.broadcast_in_dim %arg424, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %621 = stablehlo.broadcast_in_dim %arg425, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %622 = stablehlo.convert %619 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
    %623 = stablehlo.broadcast_in_dim %620, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %624 = stablehlo.subtract %622, %623 : tensor<256x14x14x256xf32>
    %625 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %626 = stablehlo.add %621, %625 : tensor<1x1x1x256xf32>
    %627 = stablehlo.rsqrt %626 : tensor<1x1x1x256xf32>
    %628 = stablehlo.reshape %arg151 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %629 = stablehlo.multiply %627, %628 : tensor<1x1x1x256xf32>
    %630 = stablehlo.broadcast_in_dim %629, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %631 = stablehlo.multiply %624, %630 : tensor<256x14x14x256xf32>
    %632 = stablehlo.reshape %arg150 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %633 = stablehlo.broadcast_in_dim %632, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %634 = stablehlo.add %631, %633 : tensor<256x14x14x256xf32>
    %635 = stablehlo.convert %634 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
    %636 = call @relu_6(%635) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
    %637 = stablehlo.convert %arg156 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
    %638 = stablehlo.convolution(%636, %637) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
    %639 = stablehlo.broadcast_in_dim %arg426, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %640 = stablehlo.broadcast_in_dim %arg427, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %641 = stablehlo.convert %638 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
    %642 = stablehlo.broadcast_in_dim %639, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %643 = stablehlo.subtract %641, %642 : tensor<256x14x14x1024xf32>
    %644 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %645 = stablehlo.add %640, %644 : tensor<1x1x1x1024xf32>
    %646 = stablehlo.rsqrt %645 : tensor<1x1x1x1024xf32>
    %647 = stablehlo.reshape %arg153 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %648 = stablehlo.multiply %646, %647 : tensor<1x1x1x1024xf32>
    %649 = stablehlo.broadcast_in_dim %648, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %650 = stablehlo.multiply %643, %649 : tensor<256x14x14x1024xf32>
    %651 = stablehlo.reshape %arg152 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %652 = stablehlo.broadcast_in_dim %651, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %653 = stablehlo.add %650, %652 : tensor<256x14x14x1024xf32>
    %654 = stablehlo.convert %653 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
    %655 = stablehlo.add %598, %654 : tensor<256x14x14x1024xf16>
    %656 = call @relu_7(%655) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
    %657 = stablehlo.convert %arg28 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
    %658 = stablehlo.convolution(%656, %657) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
    %659 = stablehlo.broadcast_in_dim %arg338, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %660 = stablehlo.broadcast_in_dim %arg339, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %661 = stablehlo.convert %658 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
    %662 = stablehlo.broadcast_in_dim %659, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %663 = stablehlo.subtract %661, %662 : tensor<256x14x14x256xf32>
    %664 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %665 = stablehlo.add %660, %664 : tensor<1x1x1x256xf32>
    %666 = stablehlo.rsqrt %665 : tensor<1x1x1x256xf32>
    %667 = stablehlo.reshape %arg23 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %668 = stablehlo.multiply %666, %667 : tensor<1x1x1x256xf32>
    %669 = stablehlo.broadcast_in_dim %668, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %670 = stablehlo.multiply %663, %669 : tensor<256x14x14x256xf32>
    %671 = stablehlo.reshape %arg22 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %672 = stablehlo.broadcast_in_dim %671, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %673 = stablehlo.add %670, %672 : tensor<256x14x14x256xf32>
    %674 = stablehlo.convert %673 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
    %675 = call @relu_6(%674) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
    %676 = stablehlo.convert %arg29 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
    %677 = stablehlo.convolution(%675, %676) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
    %678 = stablehlo.broadcast_in_dim %arg340, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %679 = stablehlo.broadcast_in_dim %arg341, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %680 = stablehlo.convert %677 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
    %681 = stablehlo.broadcast_in_dim %678, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %682 = stablehlo.subtract %680, %681 : tensor<256x14x14x256xf32>
    %683 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %684 = stablehlo.add %679, %683 : tensor<1x1x1x256xf32>
    %685 = stablehlo.rsqrt %684 : tensor<1x1x1x256xf32>
    %686 = stablehlo.reshape %arg25 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %687 = stablehlo.multiply %685, %686 : tensor<1x1x1x256xf32>
    %688 = stablehlo.broadcast_in_dim %687, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %689 = stablehlo.multiply %682, %688 : tensor<256x14x14x256xf32>
    %690 = stablehlo.reshape %arg24 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %691 = stablehlo.broadcast_in_dim %690, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %692 = stablehlo.add %689, %691 : tensor<256x14x14x256xf32>
    %693 = stablehlo.convert %692 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
    %694 = call @relu_6(%693) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
    %695 = stablehlo.convert %arg30 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
    %696 = stablehlo.convolution(%694, %695) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
    %697 = stablehlo.broadcast_in_dim %arg342, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %698 = stablehlo.broadcast_in_dim %arg343, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %699 = stablehlo.convert %696 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
    %700 = stablehlo.broadcast_in_dim %697, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %701 = stablehlo.subtract %699, %700 : tensor<256x14x14x1024xf32>
    %702 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %703 = stablehlo.add %698, %702 : tensor<1x1x1x1024xf32>
    %704 = stablehlo.rsqrt %703 : tensor<1x1x1x1024xf32>
    %705 = stablehlo.reshape %arg27 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %706 = stablehlo.multiply %704, %705 : tensor<1x1x1x1024xf32>
    %707 = stablehlo.broadcast_in_dim %706, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %708 = stablehlo.multiply %701, %707 : tensor<256x14x14x1024xf32>
    %709 = stablehlo.reshape %arg26 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %710 = stablehlo.broadcast_in_dim %709, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %711 = stablehlo.add %708, %710 : tensor<256x14x14x1024xf32>
    %712 = stablehlo.convert %711 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
    %713 = stablehlo.add %656, %712 : tensor<256x14x14x1024xf16>
    %714 = call @relu_7(%713) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
    %715 = stablehlo.convert %arg37 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
    %716 = stablehlo.convolution(%714, %715) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
    %717 = stablehlo.broadcast_in_dim %arg344, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %718 = stablehlo.broadcast_in_dim %arg345, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %719 = stablehlo.convert %716 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
    %720 = stablehlo.broadcast_in_dim %717, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %721 = stablehlo.subtract %719, %720 : tensor<256x14x14x256xf32>
    %722 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %723 = stablehlo.add %718, %722 : tensor<1x1x1x256xf32>
    %724 = stablehlo.rsqrt %723 : tensor<1x1x1x256xf32>
    %725 = stablehlo.reshape %arg32 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %726 = stablehlo.multiply %724, %725 : tensor<1x1x1x256xf32>
    %727 = stablehlo.broadcast_in_dim %726, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %728 = stablehlo.multiply %721, %727 : tensor<256x14x14x256xf32>
    %729 = stablehlo.reshape %arg31 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %730 = stablehlo.broadcast_in_dim %729, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %731 = stablehlo.add %728, %730 : tensor<256x14x14x256xf32>
    %732 = stablehlo.convert %731 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
    %733 = call @relu_6(%732) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
    %734 = stablehlo.convert %arg38 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
    %735 = stablehlo.convolution(%733, %734) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
    %736 = stablehlo.broadcast_in_dim %arg346, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %737 = stablehlo.broadcast_in_dim %arg347, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %738 = stablehlo.convert %735 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
    %739 = stablehlo.broadcast_in_dim %736, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %740 = stablehlo.subtract %738, %739 : tensor<256x14x14x256xf32>
    %741 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %742 = stablehlo.add %737, %741 : tensor<1x1x1x256xf32>
    %743 = stablehlo.rsqrt %742 : tensor<1x1x1x256xf32>
    %744 = stablehlo.reshape %arg34 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %745 = stablehlo.multiply %743, %744 : tensor<1x1x1x256xf32>
    %746 = stablehlo.broadcast_in_dim %745, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %747 = stablehlo.multiply %740, %746 : tensor<256x14x14x256xf32>
    %748 = stablehlo.reshape %arg33 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %749 = stablehlo.broadcast_in_dim %748, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %750 = stablehlo.add %747, %749 : tensor<256x14x14x256xf32>
    %751 = stablehlo.convert %750 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
    %752 = call @relu_6(%751) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
    %753 = stablehlo.convert %arg39 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
    %754 = stablehlo.convolution(%752, %753) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
    %755 = stablehlo.broadcast_in_dim %arg348, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %756 = stablehlo.broadcast_in_dim %arg349, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %757 = stablehlo.convert %754 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
    %758 = stablehlo.broadcast_in_dim %755, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %759 = stablehlo.subtract %757, %758 : tensor<256x14x14x1024xf32>
    %760 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %761 = stablehlo.add %756, %760 : tensor<1x1x1x1024xf32>
    %762 = stablehlo.rsqrt %761 : tensor<1x1x1x1024xf32>
    %763 = stablehlo.reshape %arg36 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %764 = stablehlo.multiply %762, %763 : tensor<1x1x1x1024xf32>
    %765 = stablehlo.broadcast_in_dim %764, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %766 = stablehlo.multiply %759, %765 : tensor<256x14x14x1024xf32>
    %767 = stablehlo.reshape %arg35 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %768 = stablehlo.broadcast_in_dim %767, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %769 = stablehlo.add %766, %768 : tensor<256x14x14x1024xf32>
    %770 = stablehlo.convert %769 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
    %771 = stablehlo.add %714, %770 : tensor<256x14x14x1024xf16>
    %772 = call @relu_7(%771) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
    %773 = stablehlo.convert %arg46 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
    %774 = stablehlo.convolution(%772, %773) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
    %775 = stablehlo.broadcast_in_dim %arg350, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %776 = stablehlo.broadcast_in_dim %arg351, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %777 = stablehlo.convert %774 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
    %778 = stablehlo.broadcast_in_dim %775, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %779 = stablehlo.subtract %777, %778 : tensor<256x14x14x256xf32>
    %780 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %781 = stablehlo.add %776, %780 : tensor<1x1x1x256xf32>
    %782 = stablehlo.rsqrt %781 : tensor<1x1x1x256xf32>
    %783 = stablehlo.reshape %arg41 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %784 = stablehlo.multiply %782, %783 : tensor<1x1x1x256xf32>
    %785 = stablehlo.broadcast_in_dim %784, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %786 = stablehlo.multiply %779, %785 : tensor<256x14x14x256xf32>
    %787 = stablehlo.reshape %arg40 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %788 = stablehlo.broadcast_in_dim %787, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %789 = stablehlo.add %786, %788 : tensor<256x14x14x256xf32>
    %790 = stablehlo.convert %789 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
    %791 = call @relu_6(%790) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
    %792 = stablehlo.convert %arg47 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
    %793 = stablehlo.convolution(%791, %792) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
    %794 = stablehlo.broadcast_in_dim %arg352, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %795 = stablehlo.broadcast_in_dim %arg353, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %796 = stablehlo.convert %793 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
    %797 = stablehlo.broadcast_in_dim %794, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %798 = stablehlo.subtract %796, %797 : tensor<256x14x14x256xf32>
    %799 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %800 = stablehlo.add %795, %799 : tensor<1x1x1x256xf32>
    %801 = stablehlo.rsqrt %800 : tensor<1x1x1x256xf32>
    %802 = stablehlo.reshape %arg43 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %803 = stablehlo.multiply %801, %802 : tensor<1x1x1x256xf32>
    %804 = stablehlo.broadcast_in_dim %803, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %805 = stablehlo.multiply %798, %804 : tensor<256x14x14x256xf32>
    %806 = stablehlo.reshape %arg42 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %807 = stablehlo.broadcast_in_dim %806, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %808 = stablehlo.add %805, %807 : tensor<256x14x14x256xf32>
    %809 = stablehlo.convert %808 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
    %810 = call @relu_6(%809) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
    %811 = stablehlo.convert %arg48 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
    %812 = stablehlo.convolution(%810, %811) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
    %813 = stablehlo.broadcast_in_dim %arg354, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %814 = stablehlo.broadcast_in_dim %arg355, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %815 = stablehlo.convert %812 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
    %816 = stablehlo.broadcast_in_dim %813, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %817 = stablehlo.subtract %815, %816 : tensor<256x14x14x1024xf32>
    %818 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %819 = stablehlo.add %814, %818 : tensor<1x1x1x1024xf32>
    %820 = stablehlo.rsqrt %819 : tensor<1x1x1x1024xf32>
    %821 = stablehlo.reshape %arg45 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %822 = stablehlo.multiply %820, %821 : tensor<1x1x1x1024xf32>
    %823 = stablehlo.broadcast_in_dim %822, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %824 = stablehlo.multiply %817, %823 : tensor<256x14x14x1024xf32>
    %825 = stablehlo.reshape %arg44 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %826 = stablehlo.broadcast_in_dim %825, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %827 = stablehlo.add %824, %826 : tensor<256x14x14x1024xf32>
    %828 = stablehlo.convert %827 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
    %829 = stablehlo.add %772, %828 : tensor<256x14x14x1024xf16>
    %830 = call @relu_7(%829) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
    %831 = stablehlo.convert %arg55 : (tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x512xf16>
    %832 = stablehlo.convolution(%830, %831) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x512xf16>) -> tensor<256x14x14x512xf16>
    %833 = stablehlo.broadcast_in_dim %arg356, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %834 = stablehlo.broadcast_in_dim %arg357, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %835 = stablehlo.convert %832 : (tensor<256x14x14x512xf16>) -> tensor<256x14x14x512xf32>
    %836 = stablehlo.broadcast_in_dim %833, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x14x14x512xf32>
    %837 = stablehlo.subtract %835, %836 : tensor<256x14x14x512xf32>
    %838 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %839 = stablehlo.add %834, %838 : tensor<1x1x1x512xf32>
    %840 = stablehlo.rsqrt %839 : tensor<1x1x1x512xf32>
    %841 = stablehlo.reshape %arg50 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %842 = stablehlo.multiply %840, %841 : tensor<1x1x1x512xf32>
    %843 = stablehlo.broadcast_in_dim %842, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x14x14x512xf32>
    %844 = stablehlo.multiply %837, %843 : tensor<256x14x14x512xf32>
    %845 = stablehlo.reshape %arg49 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %846 = stablehlo.broadcast_in_dim %845, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x14x14x512xf32>
    %847 = stablehlo.add %844, %846 : tensor<256x14x14x512xf32>
    %848 = stablehlo.convert %847 : (tensor<256x14x14x512xf32>) -> tensor<256x14x14x512xf16>
    %849 = call @relu_8(%848) : (tensor<256x14x14x512xf16>) -> tensor<256x14x14x512xf16>
    %850 = stablehlo.convert %arg56 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
    %851 = stablehlo.convolution(%849, %850) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
    %852 = stablehlo.broadcast_in_dim %arg358, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %853 = stablehlo.broadcast_in_dim %arg359, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %854 = stablehlo.convert %851 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
    %855 = stablehlo.broadcast_in_dim %852, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %856 = stablehlo.subtract %854, %855 : tensor<256x7x7x512xf32>
    %857 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %858 = stablehlo.add %853, %857 : tensor<1x1x1x512xf32>
    %859 = stablehlo.rsqrt %858 : tensor<1x1x1x512xf32>
    %860 = stablehlo.reshape %arg52 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %861 = stablehlo.multiply %859, %860 : tensor<1x1x1x512xf32>
    %862 = stablehlo.broadcast_in_dim %861, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %863 = stablehlo.multiply %856, %862 : tensor<256x7x7x512xf32>
    %864 = stablehlo.reshape %arg51 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %865 = stablehlo.broadcast_in_dim %864, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %866 = stablehlo.add %863, %865 : tensor<256x7x7x512xf32>
    %867 = stablehlo.convert %866 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
    %868 = call @relu_9(%867) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
    %869 = stablehlo.convert %arg57 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xf16>
    %870 = stablehlo.convolution(%868, %869) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<1x1x512x2048xf16>) -> tensor<256x7x7x2048xf16>
    %871 = stablehlo.broadcast_in_dim %arg360, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %872 = stablehlo.broadcast_in_dim %arg361, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %873 = stablehlo.convert %870 : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf32>
    %874 = stablehlo.broadcast_in_dim %871, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %875 = stablehlo.subtract %873, %874 : tensor<256x7x7x2048xf32>
    %876 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %877 = stablehlo.add %872, %876 : tensor<1x1x1x2048xf32>
    %878 = stablehlo.rsqrt %877 : tensor<1x1x1x2048xf32>
    %879 = stablehlo.reshape %arg54 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %880 = stablehlo.multiply %878, %879 : tensor<1x1x1x2048xf32>
    %881 = stablehlo.broadcast_in_dim %880, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %882 = stablehlo.multiply %875, %881 : tensor<256x7x7x2048xf32>
    %883 = stablehlo.reshape %arg53 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %884 = stablehlo.broadcast_in_dim %883, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %885 = stablehlo.add %882, %884 : tensor<256x7x7x2048xf32>
    %886 = stablehlo.convert %885 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xf16>
    %887 = stablehlo.convert %arg58 : (tensor<1x1x1024x2048xf32>) -> tensor<1x1x1024x2048xf16>
    %888 = stablehlo.convolution(%830, %887) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x2048xf16>) -> tensor<256x7x7x2048xf16>
    %889 = stablehlo.broadcast_in_dim %arg362, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %890 = stablehlo.broadcast_in_dim %arg363, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %891 = stablehlo.convert %888 : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf32>
    %892 = stablehlo.broadcast_in_dim %889, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %893 = stablehlo.subtract %891, %892 : tensor<256x7x7x2048xf32>
    %894 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %895 = stablehlo.add %890, %894 : tensor<1x1x1x2048xf32>
    %896 = stablehlo.rsqrt %895 : tensor<1x1x1x2048xf32>
    %897 = stablehlo.reshape %arg60 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %898 = stablehlo.multiply %896, %897 : tensor<1x1x1x2048xf32>
    %899 = stablehlo.broadcast_in_dim %898, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %900 = stablehlo.multiply %893, %899 : tensor<256x7x7x2048xf32>
    %901 = stablehlo.reshape %arg59 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %902 = stablehlo.broadcast_in_dim %901, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %903 = stablehlo.add %900, %902 : tensor<256x7x7x2048xf32>
    %904 = stablehlo.convert %903 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xf16>
    %905 = stablehlo.add %904, %886 : tensor<256x7x7x2048xf16>
    %906 = call @relu_10(%905) : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf16>
    %907 = stablehlo.convert %arg67 : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf16>
    %908 = stablehlo.convolution(%906, %907) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x2048xf16>, tensor<1x1x2048x512xf16>) -> tensor<256x7x7x512xf16>
    %909 = stablehlo.broadcast_in_dim %arg364, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %910 = stablehlo.broadcast_in_dim %arg365, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %911 = stablehlo.convert %908 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
    %912 = stablehlo.broadcast_in_dim %909, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %913 = stablehlo.subtract %911, %912 : tensor<256x7x7x512xf32>
    %914 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %915 = stablehlo.add %910, %914 : tensor<1x1x1x512xf32>
    %916 = stablehlo.rsqrt %915 : tensor<1x1x1x512xf32>
    %917 = stablehlo.reshape %arg62 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %918 = stablehlo.multiply %916, %917 : tensor<1x1x1x512xf32>
    %919 = stablehlo.broadcast_in_dim %918, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %920 = stablehlo.multiply %913, %919 : tensor<256x7x7x512xf32>
    %921 = stablehlo.reshape %arg61 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %922 = stablehlo.broadcast_in_dim %921, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %923 = stablehlo.add %920, %922 : tensor<256x7x7x512xf32>
    %924 = stablehlo.convert %923 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
    %925 = call @relu_9(%924) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
    %926 = stablehlo.convert %arg68 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
    %927 = stablehlo.convolution(%925, %926) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
    %928 = stablehlo.broadcast_in_dim %arg366, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %929 = stablehlo.broadcast_in_dim %arg367, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %930 = stablehlo.convert %927 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
    %931 = stablehlo.broadcast_in_dim %928, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %932 = stablehlo.subtract %930, %931 : tensor<256x7x7x512xf32>
    %933 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %934 = stablehlo.add %929, %933 : tensor<1x1x1x512xf32>
    %935 = stablehlo.rsqrt %934 : tensor<1x1x1x512xf32>
    %936 = stablehlo.reshape %arg64 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %937 = stablehlo.multiply %935, %936 : tensor<1x1x1x512xf32>
    %938 = stablehlo.broadcast_in_dim %937, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %939 = stablehlo.multiply %932, %938 : tensor<256x7x7x512xf32>
    %940 = stablehlo.reshape %arg63 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %941 = stablehlo.broadcast_in_dim %940, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %942 = stablehlo.add %939, %941 : tensor<256x7x7x512xf32>
    %943 = stablehlo.convert %942 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
    %944 = call @relu_9(%943) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
    %945 = stablehlo.convert %arg69 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xf16>
    %946 = stablehlo.convolution(%944, %945) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<1x1x512x2048xf16>) -> tensor<256x7x7x2048xf16>
    %947 = stablehlo.broadcast_in_dim %arg368, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %948 = stablehlo.broadcast_in_dim %arg369, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %949 = stablehlo.convert %946 : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf32>
    %950 = stablehlo.broadcast_in_dim %947, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %951 = stablehlo.subtract %949, %950 : tensor<256x7x7x2048xf32>
    %952 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %953 = stablehlo.add %948, %952 : tensor<1x1x1x2048xf32>
    %954 = stablehlo.rsqrt %953 : tensor<1x1x1x2048xf32>
    %955 = stablehlo.reshape %arg66 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %956 = stablehlo.multiply %954, %955 : tensor<1x1x1x2048xf32>
    %957 = stablehlo.broadcast_in_dim %956, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %958 = stablehlo.multiply %951, %957 : tensor<256x7x7x2048xf32>
    %959 = stablehlo.reshape %arg65 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %960 = stablehlo.broadcast_in_dim %959, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %961 = stablehlo.add %958, %960 : tensor<256x7x7x2048xf32>
    %962 = stablehlo.convert %961 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xf16>
    %963 = stablehlo.add %906, %962 : tensor<256x7x7x2048xf16>
    %964 = call @relu_10(%963) : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf16>
    %965 = stablehlo.convert %arg76 : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf16>
    %966 = stablehlo.convolution(%964, %965) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x2048xf16>, tensor<1x1x2048x512xf16>) -> tensor<256x7x7x512xf16>
    %967 = stablehlo.broadcast_in_dim %arg370, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %968 = stablehlo.broadcast_in_dim %arg371, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %969 = stablehlo.convert %966 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
    %970 = stablehlo.broadcast_in_dim %967, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %971 = stablehlo.subtract %969, %970 : tensor<256x7x7x512xf32>
    %972 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %973 = stablehlo.add %968, %972 : tensor<1x1x1x512xf32>
    %974 = stablehlo.rsqrt %973 : tensor<1x1x1x512xf32>
    %975 = stablehlo.reshape %arg71 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %976 = stablehlo.multiply %974, %975 : tensor<1x1x1x512xf32>
    %977 = stablehlo.broadcast_in_dim %976, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %978 = stablehlo.multiply %971, %977 : tensor<256x7x7x512xf32>
    %979 = stablehlo.reshape %arg70 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %980 = stablehlo.broadcast_in_dim %979, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %981 = stablehlo.add %978, %980 : tensor<256x7x7x512xf32>
    %982 = stablehlo.convert %981 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
    %983 = call @relu_9(%982) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
    %984 = stablehlo.convert %arg77 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
    %985 = stablehlo.convolution(%983, %984) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
    %986 = stablehlo.broadcast_in_dim %arg372, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %987 = stablehlo.broadcast_in_dim %arg373, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %988 = stablehlo.convert %985 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
    %989 = stablehlo.broadcast_in_dim %986, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %990 = stablehlo.subtract %988, %989 : tensor<256x7x7x512xf32>
    %991 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %992 = stablehlo.add %987, %991 : tensor<1x1x1x512xf32>
    %993 = stablehlo.rsqrt %992 : tensor<1x1x1x512xf32>
    %994 = stablehlo.reshape %arg73 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %995 = stablehlo.multiply %993, %994 : tensor<1x1x1x512xf32>
    %996 = stablehlo.broadcast_in_dim %995, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %997 = stablehlo.multiply %990, %996 : tensor<256x7x7x512xf32>
    %998 = stablehlo.reshape %arg72 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %999 = stablehlo.broadcast_in_dim %998, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1000 = stablehlo.add %997, %999 : tensor<256x7x7x512xf32>
    %1001 = stablehlo.convert %1000 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
    %1002 = call @relu_9(%1001) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
    %1003 = stablehlo.convert %arg78 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xf16>
    %1004 = stablehlo.convolution(%1002, %1003) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<1x1x512x2048xf16>) -> tensor<256x7x7x2048xf16>
    %1005 = stablehlo.broadcast_in_dim %arg374, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1006 = stablehlo.broadcast_in_dim %arg375, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1007 = stablehlo.convert %1004 : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf32>
    %1008 = stablehlo.broadcast_in_dim %1005, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %1009 = stablehlo.subtract %1007, %1008 : tensor<256x7x7x2048xf32>
    %1010 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %1011 = stablehlo.add %1006, %1010 : tensor<1x1x1x2048xf32>
    %1012 = stablehlo.rsqrt %1011 : tensor<1x1x1x2048xf32>
    %1013 = stablehlo.reshape %arg75 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1014 = stablehlo.multiply %1012, %1013 : tensor<1x1x1x2048xf32>
    %1015 = stablehlo.broadcast_in_dim %1014, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %1016 = stablehlo.multiply %1009, %1015 : tensor<256x7x7x2048xf32>
    %1017 = stablehlo.reshape %arg74 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1018 = stablehlo.broadcast_in_dim %1017, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %1019 = stablehlo.add %1016, %1018 : tensor<256x7x7x2048xf32>
    %1020 = stablehlo.convert %1019 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xf16>
    %1021 = stablehlo.add %964, %1020 : tensor<256x7x7x2048xf16>
    %1022 = call @relu_10(%1021) : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf16>
    %1023 = stablehlo.convert %1022 : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1024 = stablehlo.reduce(%1023 init: %cst_1) applies stablehlo.add across dimensions = [1, 2] : (tensor<256x7x7x2048xf32>, tensor<f32>) -> tensor<256x2048xf32>
    %cst_2 = stablehlo.constant dense<4.900000e+01> : tensor<f32>
    %1025 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x2048xf32>
    %1026 = stablehlo.divide %1024, %1025 : tensor<256x2048xf32>
    %1027 = stablehlo.convert %1026 : (tensor<256x2048xf32>) -> tensor<256x2048xf16>
    %1028 = stablehlo.convert %arg158 : (tensor<2048x1000xf32>) -> tensor<2048x1000xf16>
    %1029 = stablehlo.convert %arg157 : (tensor<1000xf32>) -> tensor<1000xf16>
    %1030 = stablehlo.dot_general %1027, %1028, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x2048xf16>, tensor<2048x1000xf16>) -> tensor<256x1000xf16>
    %1031 = stablehlo.reshape %1029 : (tensor<1000xf16>) -> tensor<1x1000xf16>
    %1032 = stablehlo.broadcast_in_dim %1031, dims = [0, 1] : (tensor<1x1000xf16>) -> tensor<256x1000xf16>
    %1033 = stablehlo.add %1030, %1032 : tensor<256x1000xf16>
    %1034 = stablehlo.broadcast_in_dim %1033, dims = [1, 2] : (tensor<256x1000xf16>) -> tensor<1x256x1000xf16>
    return %1034 : tensor<1x256x1000xf16>
  }
  func.func private @relu(%arg0: tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x112x112x64xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x112x112x64xf16>
    return %1 : tensor<256x112x112x64xf16>
  }
  func.func private @relu_0(%arg0: tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x56x56x64xf16>
    return %1 : tensor<256x56x56x64xf16>
  }
  func.func private @relu_1(%arg0: tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x56x56x256xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x56x56x256xf16>
    return %1 : tensor<256x56x56x256xf16>
  }
  func.func private @relu_2(%arg0: tensor<256x56x56x128xf16>) -> tensor<256x56x56x128xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x56x56x128xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x56x56x128xf16>
    return %1 : tensor<256x56x56x128xf16>
  }
  func.func private @relu_3(%arg0: tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x28x28x128xf16>
    return %1 : tensor<256x28x28x128xf16>
  }
  func.func private @relu_4(%arg0: tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x28x28x512xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x28x28x512xf16>
    return %1 : tensor<256x28x28x512xf16>
  }
  func.func private @relu_5(%arg0: tensor<256x28x28x256xf16>) -> tensor<256x28x28x256xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x28x28x256xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x28x28x256xf16>
    return %1 : tensor<256x28x28x256xf16>
  }
  func.func private @relu_6(%arg0: tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x256xf16>
    return %1 : tensor<256x14x14x256xf16>
  }
  func.func private @relu_7(%arg0: tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x14x14x1024xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x1024xf16>
    return %1 : tensor<256x14x14x1024xf16>
  }
  func.func private @relu_8(%arg0: tensor<256x14x14x512xf16>) -> tensor<256x14x14x512xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x14x14x512xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x512xf16>
    return %1 : tensor<256x14x14x512xf16>
  }
  func.func private @relu_9(%arg0: tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x7x7x512xf16>
    return %1 : tensor<256x7x7x512xf16>
  }
  func.func private @relu_10(%arg0: tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x7x7x2048xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x7x7x2048xf16>
    return %1 : tensor<256x7x7x2048xf16>
  }
}
