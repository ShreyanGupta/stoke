  .text
  .globl _ZNSt8ios_base17_M_call_callbacksENS_5eventE
  .type _ZNSt8ios_base17_M_call_callbacksENS_5eventE, @function

#! file-offset 0x127d20
#! rip-offset  0xe7d20
#! capacity    288 bytes

# Text                                          #  Line  RIP      Bytes  Opcode              
._ZNSt8ios_base17_M_call_callbacksENS_5eventE:  #        0xe7d20  0      OPC=<label>         
  pushq %r13                                    #  1     0xe7d20  2      OPC=pushq_r64_1     
  movl %esi, %r13d                              #  2     0xe7d22  3      OPC=movl_r32_r32    
  pushq %r12                                    #  3     0xe7d25  2      OPC=pushq_r64_1     
  movl %edi, %r12d                              #  4     0xe7d27  3      OPC=movl_r32_r32    
  pushq %rbx                                    #  5     0xe7d2a  1      OPC=pushq_r64_1     
  movl %r12d, %r12d                             #  6     0xe7d2b  3      OPC=movl_r32_r32    
  movl 0x18(%r15,%r12,1), %ebx                  #  7     0xe7d2e  5      OPC=movl_r32_m32    
  testq %rbx, %rbx                              #  8     0xe7d33  3      OPC=testq_r64_r64   
  je .L_e7d80                                   #  9     0xe7d36  2      OPC=je_label        
  nop                                           #  10    0xe7d38  1      OPC=nop             
  nop                                           #  11    0xe7d39  1      OPC=nop             
  nop                                           #  12    0xe7d3a  1      OPC=nop             
  nop                                           #  13    0xe7d3b  1      OPC=nop             
  nop                                           #  14    0xe7d3c  1      OPC=nop             
  nop                                           #  15    0xe7d3d  1      OPC=nop             
  nop                                           #  16    0xe7d3e  1      OPC=nop             
  nop                                           #  17    0xe7d3f  1      OPC=nop             
.L_e7d40:                                       #        0xe7d40  0      OPC=<label>         
  movl %ebx, %ebx                               #  18    0xe7d40  2      OPC=movl_r32_r32    
  movl 0x4(%r15,%rbx,1), %eax                   #  19    0xe7d42  5      OPC=movl_r32_m32    
  movl %ebx, %ebx                               #  20    0xe7d47  2      OPC=movl_r32_r32    
  movl 0x8(%r15,%rbx,1), %edx                   #  21    0xe7d49  5      OPC=movl_r32_m32    
  movl %r12d, %esi                              #  22    0xe7d4e  3      OPC=movl_r32_r32    
  movl %r13d, %edi                              #  23    0xe7d51  3      OPC=movl_r32_r32    
  nop                                           #  24    0xe7d54  1      OPC=nop             
  nop                                           #  25    0xe7d55  1      OPC=nop             
  nop                                           #  26    0xe7d56  1      OPC=nop             
  nop                                           #  27    0xe7d57  1      OPC=nop             
  andl $0xffffffe0, %eax                        #  28    0xe7d58  6      OPC=andl_r32_imm32  
  nop                                           #  29    0xe7d5e  1      OPC=nop             
  nop                                           #  30    0xe7d5f  1      OPC=nop             
  nop                                           #  31    0xe7d60  1      OPC=nop             
  addq %r15, %rax                               #  32    0xe7d61  3      OPC=addq_r64_r64    
  callq %rax                                    #  33    0xe7d64  2      OPC=callq_r64       
.L_e7d60:                                       #        0xe7d66  0      OPC=<label>         
  movl %ebx, %ebx                               #  34    0xe7d66  2      OPC=movl_r32_r32    
  movl (%r15,%rbx,1), %ebx                      #  35    0xe7d68  4      OPC=movl_r32_m32    
  testq %rbx, %rbx                              #  36    0xe7d6c  3      OPC=testq_r64_r64   
  jne .L_e7d40                                  #  37    0xe7d6f  2      OPC=jne_label       
  nop                                           #  38    0xe7d71  1      OPC=nop             
  nop                                           #  39    0xe7d72  1      OPC=nop             
  nop                                           #  40    0xe7d73  1      OPC=nop             
  nop                                           #  41    0xe7d74  1      OPC=nop             
  nop                                           #  42    0xe7d75  1      OPC=nop             
  nop                                           #  43    0xe7d76  1      OPC=nop             
  nop                                           #  44    0xe7d77  1      OPC=nop             
  nop                                           #  45    0xe7d78  1      OPC=nop             
  nop                                           #  46    0xe7d79  1      OPC=nop             
  nop                                           #  47    0xe7d7a  1      OPC=nop             
  nop                                           #  48    0xe7d7b  1      OPC=nop             
  nop                                           #  49    0xe7d7c  1      OPC=nop             
  nop                                           #  50    0xe7d7d  1      OPC=nop             
  nop                                           #  51    0xe7d7e  1      OPC=nop             
  nop                                           #  52    0xe7d7f  1      OPC=nop             
  nop                                           #  53    0xe7d80  1      OPC=nop             
  nop                                           #  54    0xe7d81  1      OPC=nop             
  nop                                           #  55    0xe7d82  1      OPC=nop             
  nop                                           #  56    0xe7d83  1      OPC=nop             
  nop                                           #  57    0xe7d84  1      OPC=nop             
  nop                                           #  58    0xe7d85  1      OPC=nop             
.L_e7d80:                                       #        0xe7d86  0      OPC=<label>         
  popq %rbx                                     #  59    0xe7d86  1      OPC=popq_r64_1      
  popq %r12                                     #  60    0xe7d87  2      OPC=popq_r64_1      
  popq %r13                                     #  61    0xe7d89  2      OPC=popq_r64_1      
  popq %r11                                     #  62    0xe7d8b  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d                       #  63    0xe7d8d  7      OPC=andl_r32_imm32  
  nop                                           #  64    0xe7d94  1      OPC=nop             
  nop                                           #  65    0xe7d95  1      OPC=nop             
  nop                                           #  66    0xe7d96  1      OPC=nop             
  nop                                           #  67    0xe7d97  1      OPC=nop             
  addq %r15, %r11                               #  68    0xe7d98  3      OPC=addq_r64_r64    
  jmpq %r11                                     #  69    0xe7d9b  3      OPC=jmpq_r64        
  nop                                           #  70    0xe7d9e  1      OPC=nop             
  nop                                           #  71    0xe7d9f  1      OPC=nop             
  nop                                           #  72    0xe7da0  1      OPC=nop             
  nop                                           #  73    0xe7da1  1      OPC=nop             
  nop                                           #  74    0xe7da2  1      OPC=nop             
  nop                                           #  75    0xe7da3  1      OPC=nop             
  nop                                           #  76    0xe7da4  1      OPC=nop             
  nop                                           #  77    0xe7da5  1      OPC=nop             
  nop                                           #  78    0xe7da6  1      OPC=nop             
  nop                                           #  79    0xe7da7  1      OPC=nop             
  nop                                           #  80    0xe7da8  1      OPC=nop             
  nop                                           #  81    0xe7da9  1      OPC=nop             
  nop                                           #  82    0xe7daa  1      OPC=nop             
  nop                                           #  83    0xe7dab  1      OPC=nop             
  nop                                           #  84    0xe7dac  1      OPC=nop             
  movl %eax, %edi                               #  85    0xe7dad  2      OPC=movl_r32_r32    
  nop                                           #  86    0xe7daf  1      OPC=nop             
  nop                                           #  87    0xe7db0  1      OPC=nop             
  nop                                           #  88    0xe7db1  1      OPC=nop             
  nop                                           #  89    0xe7db2  1      OPC=nop             
  nop                                           #  90    0xe7db3  1      OPC=nop             
  nop                                           #  91    0xe7db4  1      OPC=nop             
  nop                                           #  92    0xe7db5  1      OPC=nop             
  nop                                           #  93    0xe7db6  1      OPC=nop             
  nop                                           #  94    0xe7db7  1      OPC=nop             
  nop                                           #  95    0xe7db8  1      OPC=nop             
  nop                                           #  96    0xe7db9  1      OPC=nop             
  nop                                           #  97    0xe7dba  1      OPC=nop             
  nop                                           #  98    0xe7dbb  1      OPC=nop             
  nop                                           #  99    0xe7dbc  1      OPC=nop             
  nop                                           #  100   0xe7dbd  1      OPC=nop             
  nop                                           #  101   0xe7dbe  1      OPC=nop             
  nop                                           #  102   0xe7dbf  1      OPC=nop             
  nop                                           #  103   0xe7dc0  1      OPC=nop             
  nop                                           #  104   0xe7dc1  1      OPC=nop             
  nop                                           #  105   0xe7dc2  1      OPC=nop             
  nop                                           #  106   0xe7dc3  1      OPC=nop             
  nop                                           #  107   0xe7dc4  1      OPC=nop             
  nop                                           #  108   0xe7dc5  1      OPC=nop             
  nop                                           #  109   0xe7dc6  1      OPC=nop             
  nop                                           #  110   0xe7dc7  1      OPC=nop             
  callq .__cxa_begin_catch                      #  111   0xe7dc8  5      OPC=callq_label     
  nop                                           #  112   0xe7dcd  1      OPC=nop             
  nop                                           #  113   0xe7dce  1      OPC=nop             
  nop                                           #  114   0xe7dcf  1      OPC=nop             
  nop                                           #  115   0xe7dd0  1      OPC=nop             
  nop                                           #  116   0xe7dd1  1      OPC=nop             
  nop                                           #  117   0xe7dd2  1      OPC=nop             
  nop                                           #  118   0xe7dd3  1      OPC=nop             
  nop                                           #  119   0xe7dd4  1      OPC=nop             
  nop                                           #  120   0xe7dd5  1      OPC=nop             
  nop                                           #  121   0xe7dd6  1      OPC=nop             
  nop                                           #  122   0xe7dd7  1      OPC=nop             
  nop                                           #  123   0xe7dd8  1      OPC=nop             
  nop                                           #  124   0xe7dd9  1      OPC=nop             
  nop                                           #  125   0xe7dda  1      OPC=nop             
  nop                                           #  126   0xe7ddb  1      OPC=nop             
  nop                                           #  127   0xe7ddc  1      OPC=nop             
  nop                                           #  128   0xe7ddd  1      OPC=nop             
  nop                                           #  129   0xe7dde  1      OPC=nop             
  nop                                           #  130   0xe7ddf  1      OPC=nop             
  nop                                           #  131   0xe7de0  1      OPC=nop             
  nop                                           #  132   0xe7de1  1      OPC=nop             
  nop                                           #  133   0xe7de2  1      OPC=nop             
  nop                                           #  134   0xe7de3  1      OPC=nop             
  nop                                           #  135   0xe7de4  1      OPC=nop             
  nop                                           #  136   0xe7de5  1      OPC=nop             
  nop                                           #  137   0xe7de6  1      OPC=nop             
  nop                                           #  138   0xe7de7  1      OPC=nop             
  callq .__cxa_end_catch                        #  139   0xe7de8  5      OPC=callq_label     
  jmpq .L_e7d60                                 #  140   0xe7ded  5      OPC=jmpq_label_1    
  nop                                           #  141   0xe7df2  1      OPC=nop             
  nop                                           #  142   0xe7df3  1      OPC=nop             
  nop                                           #  143   0xe7df4  1      OPC=nop             
  nop                                           #  144   0xe7df5  1      OPC=nop             
  nop                                           #  145   0xe7df6  1      OPC=nop             
  nop                                           #  146   0xe7df7  1      OPC=nop             
  nop                                           #  147   0xe7df8  1      OPC=nop             
  nop                                           #  148   0xe7df9  1      OPC=nop             
  nop                                           #  149   0xe7dfa  1      OPC=nop             
  nop                                           #  150   0xe7dfb  1      OPC=nop             
  nop                                           #  151   0xe7dfc  1      OPC=nop             
  nop                                           #  152   0xe7dfd  1      OPC=nop             
  nop                                           #  153   0xe7dfe  1      OPC=nop             
  nop                                           #  154   0xe7dff  1      OPC=nop             
  nop                                           #  155   0xe7e00  1      OPC=nop             
  nop                                           #  156   0xe7e01  1      OPC=nop             
  nop                                           #  157   0xe7e02  1      OPC=nop             
  nop                                           #  158   0xe7e03  1      OPC=nop             
  nop                                           #  159   0xe7e04  1      OPC=nop             
  nop                                           #  160   0xe7e05  1      OPC=nop             
  nop                                           #  161   0xe7e06  1      OPC=nop             
  nop                                           #  162   0xe7e07  1      OPC=nop             
  nop                                           #  163   0xe7e08  1      OPC=nop             
  nop                                           #  164   0xe7e09  1      OPC=nop             
  nop                                           #  165   0xe7e0a  1      OPC=nop             
  nop                                           #  166   0xe7e0b  1      OPC=nop             
  nop                                           #  167   0xe7e0c  1      OPC=nop             
  cmpq $0xff, %rdx                              #  168   0xe7e0d  4      OPC=cmpq_r64_imm8   
  movl %eax, %edi                               #  169   0xe7e11  2      OPC=movl_r32_r32    
  je .L_e7e20                                   #  170   0xe7e13  2      OPC=je_label        
  nop                                           #  171   0xe7e15  1      OPC=nop             
  nop                                           #  172   0xe7e16  1      OPC=nop             
  nop                                           #  173   0xe7e17  1      OPC=nop             
  nop                                           #  174   0xe7e18  1      OPC=nop             
  nop                                           #  175   0xe7e19  1      OPC=nop             
  nop                                           #  176   0xe7e1a  1      OPC=nop             
  nop                                           #  177   0xe7e1b  1      OPC=nop             
  nop                                           #  178   0xe7e1c  1      OPC=nop             
  nop                                           #  179   0xe7e1d  1      OPC=nop             
  nop                                           #  180   0xe7e1e  1      OPC=nop             
  nop                                           #  181   0xe7e1f  1      OPC=nop             
  nop                                           #  182   0xe7e20  1      OPC=nop             
  nop                                           #  183   0xe7e21  1      OPC=nop             
  nop                                           #  184   0xe7e22  1      OPC=nop             
  nop                                           #  185   0xe7e23  1      OPC=nop             
  nop                                           #  186   0xe7e24  1      OPC=nop             
  nop                                           #  187   0xe7e25  1      OPC=nop             
  nop                                           #  188   0xe7e26  1      OPC=nop             
  nop                                           #  189   0xe7e27  1      OPC=nop             
  callq ._Unwind_Resume                         #  190   0xe7e28  5      OPC=callq_label     
.L_e7e20:                                       #        0xe7e2d  0      OPC=<label>         
  nop                                           #  191   0xe7e2d  1      OPC=nop             
  nop                                           #  192   0xe7e2e  1      OPC=nop             
  nop                                           #  193   0xe7e2f  1      OPC=nop             
  nop                                           #  194   0xe7e30  1      OPC=nop             
  nop                                           #  195   0xe7e31  1      OPC=nop             
  nop                                           #  196   0xe7e32  1      OPC=nop             
  nop                                           #  197   0xe7e33  1      OPC=nop             
  nop                                           #  198   0xe7e34  1      OPC=nop             
  nop                                           #  199   0xe7e35  1      OPC=nop             
  nop                                           #  200   0xe7e36  1      OPC=nop             
  nop                                           #  201   0xe7e37  1      OPC=nop             
  nop                                           #  202   0xe7e38  1      OPC=nop             
  nop                                           #  203   0xe7e39  1      OPC=nop             
  nop                                           #  204   0xe7e3a  1      OPC=nop             
  nop                                           #  205   0xe7e3b  1      OPC=nop             
  nop                                           #  206   0xe7e3c  1      OPC=nop             
  nop                                           #  207   0xe7e3d  1      OPC=nop             
  nop                                           #  208   0xe7e3e  1      OPC=nop             
  nop                                           #  209   0xe7e3f  1      OPC=nop             
  nop                                           #  210   0xe7e40  1      OPC=nop             
  nop                                           #  211   0xe7e41  1      OPC=nop             
  nop                                           #  212   0xe7e42  1      OPC=nop             
  nop                                           #  213   0xe7e43  1      OPC=nop             
  nop                                           #  214   0xe7e44  1      OPC=nop             
  nop                                           #  215   0xe7e45  1      OPC=nop             
  nop                                           #  216   0xe7e46  1      OPC=nop             
  nop                                           #  217   0xe7e47  1      OPC=nop             
  callq .__cxa_call_unexpected                  #  218   0xe7e48  5      OPC=callq_label     
                                                                                             
.size _ZNSt8ios_base17_M_call_callbacksENS_5eventE, .-_ZNSt8ios_base17_M_call_callbacksENS_5eventE

