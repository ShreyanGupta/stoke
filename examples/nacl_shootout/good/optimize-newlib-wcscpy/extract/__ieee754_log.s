  .text
  .globl __ieee754_log
  .type __ieee754_log, @function

#! file-offset 0x147b80
#! rip-offset  0x107b80
#! capacity    992 bytes

# Text                            #  Line  RIP       Bytes  Opcode                 
.__ieee754_log:                   #        0x107b80  0      OPC=<label>            
  movsd %xmm0, -0x8(%rsp)         #  1     0x107b80  6      OPC=movsd_m64_xmm      
  movq -0x8(%rsp), %rdx           #  2     0x107b86  5      OPC=movq_r64_m64       
  xorl %ecx, %ecx                 #  3     0x107b8b  2      OPC=xorl_r32_r32       
  movq %rdx, %rax                 #  4     0x107b8d  3      OPC=movq_r64_r64       
  shrq $0x20, %rax                #  5     0x107b90  4      OPC=shrq_r64_imm8      
  cmpl $0xfffff, %eax             #  6     0x107b94  5      OPC=cmpl_eax_imm32     
  movl %eax, %esi                 #  7     0x107b99  2      OPC=movl_r32_r32       
  nop                             #  8     0x107b9b  1      OPC=nop                
  nop                             #  9     0x107b9c  1      OPC=nop                
  nop                             #  10    0x107b9d  1      OPC=nop                
  nop                             #  11    0x107b9e  1      OPC=nop                
  nop                             #  12    0x107b9f  1      OPC=nop                
  jg .L_107c00                    #  13    0x107ba0  2      OPC=jg_label           
  andl $0x7fffffff, %esi          #  14    0x107ba2  6      OPC=andl_r32_imm32     
  je .L_107ce0                    #  15    0x107ba8  6      OPC=je_label_1         
  nop                             #  16    0x107bae  1      OPC=nop                
  nop                             #  17    0x107baf  1      OPC=nop                
  nop                             #  18    0x107bb0  1      OPC=nop                
  nop                             #  19    0x107bb1  1      OPC=nop                
  nop                             #  20    0x107bb2  1      OPC=nop                
  nop                             #  21    0x107bb3  1      OPC=nop                
  nop                             #  22    0x107bb4  1      OPC=nop                
  nop                             #  23    0x107bb5  1      OPC=nop                
  nop                             #  24    0x107bb6  1      OPC=nop                
  nop                             #  25    0x107bb7  1      OPC=nop                
  nop                             #  26    0x107bb8  1      OPC=nop                
  nop                             #  27    0x107bb9  1      OPC=nop                
  nop                             #  28    0x107bba  1      OPC=nop                
  nop                             #  29    0x107bbb  1      OPC=nop                
  nop                             #  30    0x107bbc  1      OPC=nop                
  nop                             #  31    0x107bbd  1      OPC=nop                
  nop                             #  32    0x107bbe  1      OPC=nop                
  nop                             #  33    0x107bbf  1      OPC=nop                
.L_107bc0:                        #        0x107bc0  0      OPC=<label>            
  testl %eax, %eax                #  34    0x107bc0  2      OPC=testl_r32_r32      
  js .L_107ee0                    #  35    0x107bc2  6      OPC=js_label_1         
  mulsd 0xff38630(%rip), %xmm0    #  36    0x107bc8  8      OPC=mulsd_xmm_m64      
  movl $0xffffffca, %ecx          #  37    0x107bd0  6      OPC=movl_r32_imm32_1   
  movsd %xmm0, -0x8(%rsp)         #  38    0x107bd6  6      OPC=movsd_m64_xmm      
  movq -0x8(%rsp), %rax           #  39    0x107bdc  5      OPC=movq_r64_m64       
  shrq $0x20, %rax                #  40    0x107be1  4      OPC=shrq_r64_imm8      
  nop                             #  41    0x107be5  1      OPC=nop                
  nop                             #  42    0x107be6  1      OPC=nop                
  nop                             #  43    0x107be7  1      OPC=nop                
  nop                             #  44    0x107be8  1      OPC=nop                
  nop                             #  45    0x107be9  1      OPC=nop                
  nop                             #  46    0x107bea  1      OPC=nop                
  nop                             #  47    0x107beb  1      OPC=nop                
  nop                             #  48    0x107bec  1      OPC=nop                
  nop                             #  49    0x107bed  1      OPC=nop                
  nop                             #  50    0x107bee  1      OPC=nop                
  nop                             #  51    0x107bef  1      OPC=nop                
  nop                             #  52    0x107bf0  1      OPC=nop                
  nop                             #  53    0x107bf1  1      OPC=nop                
  nop                             #  54    0x107bf2  1      OPC=nop                
  nop                             #  55    0x107bf3  1      OPC=nop                
  nop                             #  56    0x107bf4  1      OPC=nop                
  nop                             #  57    0x107bf5  1      OPC=nop                
  nop                             #  58    0x107bf6  1      OPC=nop                
  nop                             #  59    0x107bf7  1      OPC=nop                
  nop                             #  60    0x107bf8  1      OPC=nop                
  nop                             #  61    0x107bf9  1      OPC=nop                
  nop                             #  62    0x107bfa  1      OPC=nop                
  nop                             #  63    0x107bfb  1      OPC=nop                
  nop                             #  64    0x107bfc  1      OPC=nop                
  nop                             #  65    0x107bfd  1      OPC=nop                
  nop                             #  66    0x107bfe  1      OPC=nop                
  nop                             #  67    0x107bff  1      OPC=nop                
  nop                             #  68    0x107c00  1      OPC=nop                
.L_107c00:                        #        0x107c01  0      OPC=<label>            
  cmpl $0x7fefffff, %eax          #  69    0x107c01  5      OPC=cmpl_eax_imm32     
  jg .L_107cc0                    #  70    0x107c06  6      OPC=jg_label_1         
  movl %eax, %edi                 #  71    0x107c0c  2      OPC=movl_r32_r32       
  andl $0xfffff, %eax             #  72    0x107c0e  5      OPC=andl_eax_imm32     
  movsd %xmm0, -0x8(%rsp)         #  73    0x107c13  6      OPC=movsd_m64_xmm      
  leal 0x95f64(%rax), %edx        #  74    0x107c19  6      OPC=leal_r32_m16       
  xchgw %ax, %ax                  #  75    0x107c1f  2      OPC=xchgw_ax_r16       
  sarl $0x14, %edi                #  76    0x107c21  3      OPC=sarl_r32_imm8      
  leal -0x3ff(%rcx,%rdi,1), %edi  #  77    0x107c24  7      OPC=leal_r32_m16       
  movq -0x8(%rsp), %rcx           #  78    0x107c2b  5      OPC=movq_r64_m64       
  andl $0x100000, %edx            #  79    0x107c30  6      OPC=andl_r32_imm32     
  movl %edx, %esi                 #  80    0x107c36  2      OPC=movl_r32_r32       
  sarl $0x14, %edx                #  81    0x107c38  3      OPC=sarl_r32_imm8      
  xorl $0x3ff00000, %esi          #  82    0x107c3b  6      OPC=xorl_r32_imm32     
  andl $0xffffffff, %ecx          #  83    0x107c41  6      OPC=andl_r32_imm32     
  nop                             #  84    0x107c47  1      OPC=nop                
  nop                             #  85    0x107c48  1      OPC=nop                
  nop                             #  86    0x107c49  1      OPC=nop                
  leal (%rdi,%rdx,1), %edx        #  87    0x107c4a  3      OPC=leal_r32_m16       
  orl %eax, %esi                  #  88    0x107c4d  2      OPC=orl_r32_r32        
  shlq $0x20, %rsi                #  89    0x107c4f  4      OPC=shlq_r64_imm8      
  orq %rsi, %rcx                  #  90    0x107c53  3      OPC=orq_r64_r64        
  movq %rcx, -0x8(%rsp)           #  91    0x107c56  5      OPC=movq_m64_r64       
  leal 0x2(%rax), %ecx            #  92    0x107c5b  3      OPC=leal_r32_m16       
  movsd -0x8(%rsp), %xmm2         #  93    0x107c5e  6      OPC=movsd_xmm_m64      
  nop                             #  94    0x107c64  1      OPC=nop                
  nop                             #  95    0x107c65  1      OPC=nop                
  nop                             #  96    0x107c66  1      OPC=nop                
  andl $0xfffff, %ecx             #  97    0x107c67  6      OPC=andl_r32_imm32     
  cmpl $0x2, %ecx                 #  98    0x107c6d  3      OPC=cmpl_r32_imm8      
  subsd 0xff38597(%rip), %xmm2    #  99    0x107c70  8      OPC=subsd_xmm_m64      
  jg .L_107d20                    #  100   0x107c78  6      OPC=jg_label_1         
  xorpd %xmm0, %xmm0              #  101   0x107c7e  4      OPC=xorpd_xmm_xmm      
  ucomisd %xmm0, %xmm2            #  102   0x107c82  4      OPC=ucomisd_xmm_xmm    
  nop                             #  103   0x107c86  1      OPC=nop                
  jne .L_107e00                   #  104   0x107c87  6      OPC=jne_label_1        
  jp .L_107e00                    #  105   0x107c8d  6      OPC=jp_label_1         
  testl %edx, %edx                #  106   0x107c93  2      OPC=testl_r32_r32      
  je .L_107e80                    #  107   0x107c95  6      OPC=je_label_1         
  cvtsi2sdl %edx, %xmm1           #  108   0x107c9b  4      OPC=cvtsi2sdl_xmm_r32  
  movsd 0xff38570(%rip), %xmm0    #  109   0x107c9f  8      OPC=movsd_xmm_m64      
  mulsd %xmm1, %xmm0              #  110   0x107ca7  4      OPC=mulsd_xmm_xmm      
  mulsd 0xff3856c(%rip), %xmm1    #  111   0x107cab  8      OPC=mulsd_xmm_m64      
  addsd %xmm1, %xmm0              #  112   0x107cb3  4      OPC=addsd_xmm_xmm      
  popq %r11                       #  113   0x107cb7  2      OPC=popq_r64_1         
  andl $0xffffffe0, %r11d         #  114   0x107cb9  7      OPC=andl_r32_imm32     
  nop                             #  115   0x107cc0  1      OPC=nop                
  nop                             #  116   0x107cc1  1      OPC=nop                
  nop                             #  117   0x107cc2  1      OPC=nop                
  nop                             #  118   0x107cc3  1      OPC=nop                
  addq %r15, %r11                 #  119   0x107cc4  3      OPC=addq_r64_r64       
  jmpq %r11                       #  120   0x107cc7  3      OPC=jmpq_r64           
  nop                             #  121   0x107cca  1      OPC=nop                
  nop                             #  122   0x107ccb  1      OPC=nop                
  nop                             #  123   0x107ccc  1      OPC=nop                
  nop                             #  124   0x107ccd  1      OPC=nop                
.L_107cc0:                        #        0x107cce  0      OPC=<label>            
  addsd %xmm0, %xmm0              #  125   0x107cce  4      OPC=addsd_xmm_xmm      
  popq %r11                       #  126   0x107cd2  2      OPC=popq_r64_1         
  andl $0xffffffe0, %r11d         #  127   0x107cd4  7      OPC=andl_r32_imm32     
  nop                             #  128   0x107cdb  1      OPC=nop                
  nop                             #  129   0x107cdc  1      OPC=nop                
  nop                             #  130   0x107cdd  1      OPC=nop                
  nop                             #  131   0x107cde  1      OPC=nop                
  addq %r15, %r11                 #  132   0x107cdf  3      OPC=addq_r64_r64       
  jmpq %r11                       #  133   0x107ce2  3      OPC=jmpq_r64           
  nop                             #  134   0x107ce5  1      OPC=nop                
  nop                             #  135   0x107ce6  1      OPC=nop                
  nop                             #  136   0x107ce7  1      OPC=nop                
  nop                             #  137   0x107ce8  1      OPC=nop                
  nop                             #  138   0x107ce9  1      OPC=nop                
  nop                             #  139   0x107cea  1      OPC=nop                
  nop                             #  140   0x107ceb  1      OPC=nop                
  nop                             #  141   0x107cec  1      OPC=nop                
  nop                             #  142   0x107ced  1      OPC=nop                
  nop                             #  143   0x107cee  1      OPC=nop                
  nop                             #  144   0x107cef  1      OPC=nop                
  nop                             #  145   0x107cf0  1      OPC=nop                
  nop                             #  146   0x107cf1  1      OPC=nop                
  nop                             #  147   0x107cf2  1      OPC=nop                
  nop                             #  148   0x107cf3  1      OPC=nop                
  nop                             #  149   0x107cf4  1      OPC=nop                
.L_107ce0:                        #        0x107cf5  0      OPC=<label>            
  testl %edx, %edx                #  150   0x107cf5  2      OPC=testl_r32_r32      
  jne .L_107bc0                   #  151   0x107cf7  6      OPC=jne_label_1        
  movsd 0xff38508(%rip), %xmm0    #  152   0x107cfd  8      OPC=movsd_xmm_m64      
  divsd 0xff384f0(%rip), %xmm0    #  153   0x107d05  8      OPC=divsd_xmm_m64      
  popq %r11                       #  154   0x107d0d  2      OPC=popq_r64_1         
  nop                             #  155   0x107d0f  1      OPC=nop                
  nop                             #  156   0x107d10  1      OPC=nop                
  nop                             #  157   0x107d11  1      OPC=nop                
  nop                             #  158   0x107d12  1      OPC=nop                
  nop                             #  159   0x107d13  1      OPC=nop                
  nop                             #  160   0x107d14  1      OPC=nop                
  andl $0xffffffe0, %r11d         #  161   0x107d15  7      OPC=andl_r32_imm32     
  nop                             #  162   0x107d1c  1      OPC=nop                
  nop                             #  163   0x107d1d  1      OPC=nop                
  nop                             #  164   0x107d1e  1      OPC=nop                
  nop                             #  165   0x107d1f  1      OPC=nop                
  addq %r15, %r11                 #  166   0x107d20  3      OPC=addq_r64_r64       
  jmpq %r11                       #  167   0x107d23  3      OPC=jmpq_r64           
  nop                             #  168   0x107d26  1      OPC=nop                
  nop                             #  169   0x107d27  1      OPC=nop                
  nop                             #  170   0x107d28  1      OPC=nop                
  nop                             #  171   0x107d29  1      OPC=nop                
  nop                             #  172   0x107d2a  1      OPC=nop                
  nop                             #  173   0x107d2b  1      OPC=nop                
  nop                             #  174   0x107d2c  1      OPC=nop                
  nop                             #  175   0x107d2d  1      OPC=nop                
  nop                             #  176   0x107d2e  1      OPC=nop                
  nop                             #  177   0x107d2f  1      OPC=nop                
  nop                             #  178   0x107d30  1      OPC=nop                
  nop                             #  179   0x107d31  1      OPC=nop                
  nop                             #  180   0x107d32  1      OPC=nop                
  nop                             #  181   0x107d33  1      OPC=nop                
  nop                             #  182   0x107d34  1      OPC=nop                
  nop                             #  183   0x107d35  1      OPC=nop                
  nop                             #  184   0x107d36  1      OPC=nop                
  nop                             #  185   0x107d37  1      OPC=nop                
  nop                             #  186   0x107d38  1      OPC=nop                
  nop                             #  187   0x107d39  1      OPC=nop                
  nop                             #  188   0x107d3a  1      OPC=nop                
  nop                             #  189   0x107d3b  1      OPC=nop                
.L_107d20:                        #        0x107d3c  0      OPC=<label>            
  movsd 0xff38508(%rip), %xmm4    #  190   0x107d3c  8      OPC=movsd_xmm_m64      
  leal -0x6147a(%rax), %esi       #  191   0x107d44  6      OPC=leal_r32_m16       
  movapd %xmm2, %xmm0             #  192   0x107d4a  4      OPC=movapd_xmm_xmm     
  movl $0x6b851, %ecx             #  193   0x107d4e  5      OPC=movl_r32_imm32     
  addsd %xmm2, %xmm4              #  194   0x107d53  4      OPC=addsd_xmm_xmm      
  nop                             #  195   0x107d57  1      OPC=nop                
  nop                             #  196   0x107d58  1      OPC=nop                
  nop                             #  197   0x107d59  1      OPC=nop                
  nop                             #  198   0x107d5a  1      OPC=nop                
  nop                             #  199   0x107d5b  1      OPC=nop                
  movsd 0xff384f0(%rip), %xmm1    #  200   0x107d5c  8      OPC=movsd_xmm_m64      
  subl %eax, %ecx                 #  201   0x107d64  2      OPC=subl_r32_r32       
  cvtsi2sdl %edx, %xmm5           #  202   0x107d66  4      OPC=cvtsi2sdl_xmm_r32  
  orl %esi, %ecx                  #  203   0x107d6a  2      OPC=orl_r32_r32        
  divsd %xmm4, %xmm0              #  204   0x107d6c  4      OPC=divsd_xmm_xmm      
  movapd %xmm0, %xmm4             #  205   0x107d70  4      OPC=movapd_xmm_xmm     
  mulsd %xmm0, %xmm0              #  206   0x107d74  4      OPC=mulsd_xmm_xmm      
  movapd %xmm0, %xmm3             #  207   0x107d78  4      OPC=movapd_xmm_xmm     
  mulsd %xmm0, %xmm3              #  208   0x107d7c  4      OPC=mulsd_xmm_xmm      
  mulsd %xmm3, %xmm1              #  209   0x107d80  4      OPC=mulsd_xmm_xmm      
  addsd 0xff384d0(%rip), %xmm1    #  210   0x107d84  8      OPC=addsd_xmm_m64      
  mulsd %xmm3, %xmm1              #  211   0x107d8c  4      OPC=mulsd_xmm_xmm      
  addsd 0xff384cc(%rip), %xmm1    #  212   0x107d90  8      OPC=addsd_xmm_m64      
  mulsd %xmm3, %xmm1              #  213   0x107d98  4      OPC=mulsd_xmm_xmm      
  addsd 0xff384c8(%rip), %xmm1    #  214   0x107d9c  8      OPC=addsd_xmm_m64      
  mulsd %xmm0, %xmm1              #  215   0x107da4  4      OPC=mulsd_xmm_xmm      
  movsd 0xff384c4(%rip), %xmm0    #  216   0x107da8  8      OPC=movsd_xmm_m64      
  mulsd %xmm3, %xmm0              #  217   0x107db0  4      OPC=mulsd_xmm_xmm      
  addsd 0xff384c0(%rip), %xmm0    #  218   0x107db4  8      OPC=addsd_xmm_m64      
  mulsd %xmm3, %xmm0              #  219   0x107dbc  4      OPC=mulsd_xmm_xmm      
  addsd 0xff384bc(%rip), %xmm0    #  220   0x107dc0  8      OPC=addsd_xmm_m64      
  mulsd %xmm3, %xmm0              #  221   0x107dc8  4      OPC=mulsd_xmm_xmm      
  addsd %xmm0, %xmm1              #  222   0x107dcc  4      OPC=addsd_xmm_xmm      
  jle .L_107e40                   #  223   0x107dd0  6      OPC=jle_label_1        
  nop                             #  224   0x107dd6  1      OPC=nop                
  nop                             #  225   0x107dd7  1      OPC=nop                
  nop                             #  226   0x107dd8  1      OPC=nop                
  nop                             #  227   0x107dd9  1      OPC=nop                
  nop                             #  228   0x107dda  1      OPC=nop                
  nop                             #  229   0x107ddb  1      OPC=nop                
  movsd 0xff38428(%rip), %xmm3    #  230   0x107ddc  8      OPC=movsd_xmm_m64      
  testl %edx, %edx                #  231   0x107de4  2      OPC=testl_r32_r32      
  mulsd %xmm2, %xmm3              #  232   0x107de6  4      OPC=mulsd_xmm_xmm      
  mulsd %xmm2, %xmm3              #  233   0x107dea  4      OPC=mulsd_xmm_xmm      
  jne .L_107ea0                   #  234   0x107dee  6      OPC=jne_label_1        
  addsd %xmm3, %xmm1              #  235   0x107df4  4      OPC=addsd_xmm_xmm      
  movapd %xmm2, %xmm0             #  236   0x107df8  4      OPC=movapd_xmm_xmm     
  mulsd %xmm4, %xmm1              #  237   0x107dfc  4      OPC=mulsd_xmm_xmm      
  subsd %xmm1, %xmm3              #  238   0x107e00  4      OPC=subsd_xmm_xmm      
  subsd %xmm3, %xmm0              #  239   0x107e04  4      OPC=subsd_xmm_xmm      
  popq %r11                       #  240   0x107e08  2      OPC=popq_r64_1         
  andl $0xffffffe0, %r11d         #  241   0x107e0a  7      OPC=andl_r32_imm32     
  nop                             #  242   0x107e11  1      OPC=nop                
  nop                             #  243   0x107e12  1      OPC=nop                
  nop                             #  244   0x107e13  1      OPC=nop                
  nop                             #  245   0x107e14  1      OPC=nop                
  addq %r15, %r11                 #  246   0x107e15  3      OPC=addq_r64_r64       
  jmpq %r11                       #  247   0x107e18  3      OPC=jmpq_r64           
  nop                             #  248   0x107e1b  1      OPC=nop                
  nop                             #  249   0x107e1c  1      OPC=nop                
  nop                             #  250   0x107e1d  1      OPC=nop                
  nop                             #  251   0x107e1e  1      OPC=nop                
  nop                             #  252   0x107e1f  1      OPC=nop                
  nop                             #  253   0x107e20  1      OPC=nop                
  nop                             #  254   0x107e21  1      OPC=nop                
  nop                             #  255   0x107e22  1      OPC=nop                
.L_107e00:                        #        0x107e23  0      OPC=<label>            
  movsd 0xff38418(%rip), %xmm0    #  256   0x107e23  8      OPC=movsd_xmm_m64      
  testl %edx, %edx                #  257   0x107e2b  2      OPC=testl_r32_r32      
  movapd %xmm2, %xmm1             #  258   0x107e2d  4      OPC=movapd_xmm_xmm     
  mulsd %xmm2, %xmm0              #  259   0x107e31  4      OPC=mulsd_xmm_xmm      
  mulsd %xmm2, %xmm1              #  260   0x107e35  4      OPC=mulsd_xmm_xmm      
  addsd 0xff383d2(%rip), %xmm0    #  261   0x107e39  8      OPC=addsd_xmm_m64      
  xchgw %ax, %ax                  #  262   0x107e41  2      OPC=xchgw_ax_r16       
  mulsd %xmm0, %xmm1              #  263   0x107e43  4      OPC=mulsd_xmm_xmm      
  jne .L_107f00                   #  264   0x107e47  6      OPC=jne_label_1        
  movapd %xmm2, %xmm0             #  265   0x107e4d  4      OPC=movapd_xmm_xmm     
  subsd %xmm1, %xmm0              #  266   0x107e51  4      OPC=subsd_xmm_xmm      
  popq %r11                       #  267   0x107e55  2      OPC=popq_r64_1         
  andl $0xffffffe0, %r11d         #  268   0x107e57  7      OPC=andl_r32_imm32     
  nop                             #  269   0x107e5e  1      OPC=nop                
  nop                             #  270   0x107e5f  1      OPC=nop                
  nop                             #  271   0x107e60  1      OPC=nop                
  nop                             #  272   0x107e61  1      OPC=nop                
  addq %r15, %r11                 #  273   0x107e62  3      OPC=addq_r64_r64       
  jmpq %r11                       #  274   0x107e65  3      OPC=jmpq_r64           
  xchgw %ax, %ax                  #  275   0x107e68  2      OPC=xchgw_ax_r16       
.L_107e40:                        #        0x107e6a  0      OPC=<label>            
  testl %edx, %edx                #  276   0x107e6a  2      OPC=testl_r32_r32      
  je .L_107f40                    #  277   0x107e6c  6      OPC=je_label_1         
  movapd %xmm2, %xmm3             #  278   0x107e72  4      OPC=movapd_xmm_xmm     
  movsd 0xff383bc(%rip), %xmm0    #  279   0x107e76  8      OPC=movsd_xmm_m64      
  subsd %xmm1, %xmm3              #  280   0x107e7e  4      OPC=subsd_xmm_xmm      
  mulsd %xmm5, %xmm0              #  281   0x107e82  4      OPC=mulsd_xmm_xmm      
  nop                             #  282   0x107e86  1      OPC=nop                
  nop                             #  283   0x107e87  1      OPC=nop                
  nop                             #  284   0x107e88  1      OPC=nop                
  nop                             #  285   0x107e89  1      OPC=nop                
  mulsd 0xff383c0(%rip), %xmm5    #  286   0x107e8a  8      OPC=mulsd_xmm_m64      
  movapd %xmm3, %xmm1             #  287   0x107e92  4      OPC=movapd_xmm_xmm     
  mulsd %xmm4, %xmm1              #  288   0x107e96  4      OPC=mulsd_xmm_xmm      
  addsd %xmm5, %xmm1              #  289   0x107e9a  4      OPC=addsd_xmm_xmm      
  subsd %xmm2, %xmm1              #  290   0x107e9e  4      OPC=subsd_xmm_xmm      
  subsd %xmm1, %xmm0              #  291   0x107ea2  4      OPC=subsd_xmm_xmm      
  nop                             #  292   0x107ea6  1      OPC=nop                
  nop                             #  293   0x107ea7  1      OPC=nop                
  nop                             #  294   0x107ea8  1      OPC=nop                
  nop                             #  295   0x107ea9  1      OPC=nop                
.L_107e80:                        #        0x107eaa  0      OPC=<label>            
  popq %r11                       #  296   0x107eaa  2      OPC=popq_r64_1         
  andl $0xffffffe0, %r11d         #  297   0x107eac  7      OPC=andl_r32_imm32     
  nop                             #  298   0x107eb3  1      OPC=nop                
  nop                             #  299   0x107eb4  1      OPC=nop                
  nop                             #  300   0x107eb5  1      OPC=nop                
  nop                             #  301   0x107eb6  1      OPC=nop                
  addq %r15, %r11                 #  302   0x107eb7  3      OPC=addq_r64_r64       
  jmpq %r11                       #  303   0x107eba  3      OPC=jmpq_r64           
  nop                             #  304   0x107ebd  1      OPC=nop                
  nop                             #  305   0x107ebe  1      OPC=nop                
  nop                             #  306   0x107ebf  1      OPC=nop                
  nop                             #  307   0x107ec0  1      OPC=nop                
  nop                             #  308   0x107ec1  1      OPC=nop                
  nop                             #  309   0x107ec2  1      OPC=nop                
  nop                             #  310   0x107ec3  1      OPC=nop                
  nop                             #  311   0x107ec4  1      OPC=nop                
  nop                             #  312   0x107ec5  1      OPC=nop                
  nop                             #  313   0x107ec6  1      OPC=nop                
  nop                             #  314   0x107ec7  1      OPC=nop                
  nop                             #  315   0x107ec8  1      OPC=nop                
  nop                             #  316   0x107ec9  1      OPC=nop                
  nop                             #  317   0x107eca  1      OPC=nop                
  nop                             #  318   0x107ecb  1      OPC=nop                
  nop                             #  319   0x107ecc  1      OPC=nop                
  nop                             #  320   0x107ecd  1      OPC=nop                
  nop                             #  321   0x107ece  1      OPC=nop                
  nop                             #  322   0x107ecf  1      OPC=nop                
  nop                             #  323   0x107ed0  1      OPC=nop                
.L_107ea0:                        #        0x107ed1  0      OPC=<label>            
  addsd %xmm3, %xmm1              #  324   0x107ed1  4      OPC=addsd_xmm_xmm      
  movsd 0xff38364(%rip), %xmm0    #  325   0x107ed5  8      OPC=movsd_xmm_m64      
  mulsd %xmm5, %xmm0              #  326   0x107edd  4      OPC=mulsd_xmm_xmm      
  mulsd 0xff38360(%rip), %xmm5    #  327   0x107ee1  8      OPC=mulsd_xmm_m64      
  mulsd %xmm4, %xmm1              #  328   0x107ee9  4      OPC=mulsd_xmm_xmm      
  addsd %xmm5, %xmm1              #  329   0x107eed  4      OPC=addsd_xmm_xmm      
  subsd %xmm1, %xmm3              #  330   0x107ef1  4      OPC=subsd_xmm_xmm      
  subsd %xmm2, %xmm3              #  331   0x107ef5  4      OPC=subsd_xmm_xmm      
  subsd %xmm3, %xmm0              #  332   0x107ef9  4      OPC=subsd_xmm_xmm      
  popq %r11                       #  333   0x107efd  2      OPC=popq_r64_1         
  andl $0xffffffe0, %r11d         #  334   0x107eff  7      OPC=andl_r32_imm32     
  nop                             #  335   0x107f06  1      OPC=nop                
  nop                             #  336   0x107f07  1      OPC=nop                
  nop                             #  337   0x107f08  1      OPC=nop                
  nop                             #  338   0x107f09  1      OPC=nop                
  addq %r15, %r11                 #  339   0x107f0a  3      OPC=addq_r64_r64       
  jmpq %r11                       #  340   0x107f0d  3      OPC=jmpq_r64           
  nop                             #  341   0x107f10  1      OPC=nop                
  nop                             #  342   0x107f11  1      OPC=nop                
  nop                             #  343   0x107f12  1      OPC=nop                
  nop                             #  344   0x107f13  1      OPC=nop                
  nop                             #  345   0x107f14  1      OPC=nop                
  nop                             #  346   0x107f15  1      OPC=nop                
  nop                             #  347   0x107f16  1      OPC=nop                
  nop                             #  348   0x107f17  1      OPC=nop                
.L_107ee0:                        #        0x107f18  0      OPC=<label>            
  subsd %xmm0, %xmm0              #  349   0x107f18  4      OPC=subsd_xmm_xmm      
  divsd 0xff382fc(%rip), %xmm0    #  350   0x107f1c  8      OPC=divsd_xmm_m64      
  popq %r11                       #  351   0x107f24  2      OPC=popq_r64_1         
  andl $0xffffffe0, %r11d         #  352   0x107f26  7      OPC=andl_r32_imm32     
  nop                             #  353   0x107f2d  1      OPC=nop                
  nop                             #  354   0x107f2e  1      OPC=nop                
  nop                             #  355   0x107f2f  1      OPC=nop                
  nop                             #  356   0x107f30  1      OPC=nop                
  addq %r15, %r11                 #  357   0x107f31  3      OPC=addq_r64_r64       
  jmpq %r11                       #  358   0x107f34  3      OPC=jmpq_r64           
  nop                             #  359   0x107f37  1      OPC=nop                
  nop                             #  360   0x107f38  1      OPC=nop                
  nop                             #  361   0x107f39  1      OPC=nop                
  nop                             #  362   0x107f3a  1      OPC=nop                
  nop                             #  363   0x107f3b  1      OPC=nop                
  nop                             #  364   0x107f3c  1      OPC=nop                
  nop                             #  365   0x107f3d  1      OPC=nop                
  nop                             #  366   0x107f3e  1      OPC=nop                
.L_107f00:                        #        0x107f3f  0      OPC=<label>            
  cvtsi2sdl %edx, %xmm3           #  367   0x107f3f  4      OPC=cvtsi2sdl_xmm_r32  
  movsd 0xff38304(%rip), %xmm0    #  368   0x107f43  8      OPC=movsd_xmm_m64      
  mulsd %xmm3, %xmm0              #  369   0x107f4b  4      OPC=mulsd_xmm_xmm      
  mulsd 0xff38310(%rip), %xmm3    #  370   0x107f4f  8      OPC=mulsd_xmm_m64      
  addsd %xmm1, %xmm3              #  371   0x107f57  4      OPC=addsd_xmm_xmm      
  subsd %xmm2, %xmm3              #  372   0x107f5b  4      OPC=subsd_xmm_xmm      
  subsd %xmm3, %xmm0              #  373   0x107f5f  4      OPC=subsd_xmm_xmm      
  popq %r11                       #  374   0x107f63  2      OPC=popq_r64_1         
  andl $0xffffffe0, %r11d         #  375   0x107f65  7      OPC=andl_r32_imm32     
  nop                             #  376   0x107f6c  1      OPC=nop                
  nop                             #  377   0x107f6d  1      OPC=nop                
  nop                             #  378   0x107f6e  1      OPC=nop                
  nop                             #  379   0x107f6f  1      OPC=nop                
  addq %r15, %r11                 #  380   0x107f70  3      OPC=addq_r64_r64       
  jmpq %r11                       #  381   0x107f73  3      OPC=jmpq_r64           
  nop                             #  382   0x107f76  1      OPC=nop                
  nop                             #  383   0x107f77  1      OPC=nop                
  nop                             #  384   0x107f78  1      OPC=nop                
  nop                             #  385   0x107f79  1      OPC=nop                
  nop                             #  386   0x107f7a  1      OPC=nop                
  nop                             #  387   0x107f7b  1      OPC=nop                
  nop                             #  388   0x107f7c  1      OPC=nop                
  nop                             #  389   0x107f7d  1      OPC=nop                
  nop                             #  390   0x107f7e  1      OPC=nop                
  nop                             #  391   0x107f7f  1      OPC=nop                
  nop                             #  392   0x107f80  1      OPC=nop                
  nop                             #  393   0x107f81  1      OPC=nop                
  nop                             #  394   0x107f82  1      OPC=nop                
  nop                             #  395   0x107f83  1      OPC=nop                
  nop                             #  396   0x107f84  1      OPC=nop                
  nop                             #  397   0x107f85  1      OPC=nop                
.L_107f40:                        #        0x107f86  0      OPC=<label>            
  movapd %xmm2, %xmm0             #  398   0x107f86  4      OPC=movapd_xmm_xmm     
  subsd %xmm1, %xmm0              #  399   0x107f8a  4      OPC=subsd_xmm_xmm      
  mulsd %xmm4, %xmm0              #  400   0x107f8e  4      OPC=mulsd_xmm_xmm      
  subsd %xmm0, %xmm2              #  401   0x107f92  4      OPC=subsd_xmm_xmm      
  movapd %xmm2, %xmm0             #  402   0x107f96  4      OPC=movapd_xmm_xmm     
  popq %r11                       #  403   0x107f9a  2      OPC=popq_r64_1         
  andl $0xffffffe0, %r11d         #  404   0x107f9c  7      OPC=andl_r32_imm32     
  nop                             #  405   0x107fa3  1      OPC=nop                
  nop                             #  406   0x107fa4  1      OPC=nop                
  nop                             #  407   0x107fa5  1      OPC=nop                
  nop                             #  408   0x107fa6  1      OPC=nop                
  addq %r15, %r11                 #  409   0x107fa7  3      OPC=addq_r64_r64       
  jmpq %r11                       #  410   0x107faa  3      OPC=jmpq_r64           
                                                                                   
.size __ieee754_log, .-__ieee754_log

