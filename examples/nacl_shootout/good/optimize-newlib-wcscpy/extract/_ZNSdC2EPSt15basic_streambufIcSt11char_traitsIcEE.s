  .text
  .globl _ZNSdC2EPSt15basic_streambufIcSt11char_traitsIcEE
  .type _ZNSdC2EPSt15basic_streambufIcSt11char_traitsIcEE, @function

#! file-offset 0x13be00
#! rip-offset  0xfbe00
#! capacity    448 bytes

# Text                                                                         #  Line  RIP      Bytes  Opcode              
._ZNSdC2EPSt15basic_streambufIcSt11char_traitsIcEE:                            #        0xfbe00  0      OPC=<label>         
  movq %r12, -0x18(%rsp)                                                       #  1     0xfbe00  5      OPC=movq_m64_r64    
  movl %esi, %r12d                                                             #  2     0xfbe05  3      OPC=movl_r32_r32    
  movq %r14, -0x8(%rsp)                                                        #  3     0xfbe08  5      OPC=movq_m64_r64    
  leal 0x4(%r12), %r14d                                                        #  4     0xfbe0d  5      OPC=leal_r32_m16    
  movq %rbx, -0x20(%rsp)                                                       #  5     0xfbe12  5      OPC=movq_m64_r64    
  movq %r13, -0x10(%rsp)                                                       #  6     0xfbe17  5      OPC=movq_m64_r64    
  nop                                                                          #  7     0xfbe1c  1      OPC=nop             
  nop                                                                          #  8     0xfbe1d  1      OPC=nop             
  nop                                                                          #  9     0xfbe1e  1      OPC=nop             
  nop                                                                          #  10    0xfbe1f  1      OPC=nop             
  subl $0x38, %esp                                                             #  11    0xfbe20  3      OPC=subl_r32_imm8   
  addq %r15, %rsp                                                              #  12    0xfbe23  3      OPC=addq_r64_r64    
  movl %edi, %ebx                                                              #  13    0xfbe26  2      OPC=movl_r32_r32    
  movl %edx, %edx                                                              #  14    0xfbe28  2      OPC=movl_r32_r32    
  movl %r14d, %r14d                                                            #  15    0xfbe2a  3      OPC=movl_r32_r32    
  movl (%r15,%r14,1), %eax                                                     #  16    0xfbe2d  4      OPC=movl_r32_m32    
  leal 0x4(%r14), %r13d                                                        #  17    0xfbe31  4      OPC=leal_r32_m16    
  movq %rdx, 0x8(%rsp)                                                         #  18    0xfbe35  5      OPC=movq_m64_r64    
  movl 0x8(%rsp), %esi                                                         #  19    0xfbe3a  4      OPC=movl_r32_m32    
  xchgw %ax, %ax                                                               #  20    0xfbe3e  2      OPC=xchgw_ax_r16    
  movl %r13d, %r13d                                                            #  21    0xfbe40  3      OPC=movl_r32_r32    
  movl (%r15,%r13,1), %edx                                                     #  22    0xfbe43  4      OPC=movl_r32_m32    
  movl %ebx, %ebx                                                              #  23    0xfbe47  2      OPC=movl_r32_r32    
  movl %eax, (%r15,%rbx,1)                                                     #  24    0xfbe49  4      OPC=movl_m32_r32    
  subl $0xc, %eax                                                              #  25    0xfbe4d  3      OPC=subl_r32_imm8   
  movl %eax, %eax                                                              #  26    0xfbe50  2      OPC=movl_r32_r32    
  movl (%r15,%rax,1), %eax                                                     #  27    0xfbe52  4      OPC=movl_r32_m32    
  nop                                                                          #  28    0xfbe56  1      OPC=nop             
  nop                                                                          #  29    0xfbe57  1      OPC=nop             
  nop                                                                          #  30    0xfbe58  1      OPC=nop             
  nop                                                                          #  31    0xfbe59  1      OPC=nop             
  nop                                                                          #  32    0xfbe5a  1      OPC=nop             
  nop                                                                          #  33    0xfbe5b  1      OPC=nop             
  nop                                                                          #  34    0xfbe5c  1      OPC=nop             
  nop                                                                          #  35    0xfbe5d  1      OPC=nop             
  nop                                                                          #  36    0xfbe5e  1      OPC=nop             
  nop                                                                          #  37    0xfbe5f  1      OPC=nop             
  movl %ebx, %ebx                                                              #  38    0xfbe60  2      OPC=movl_r32_r32    
  movl $0x0, 0x4(%r15,%rbx,1)                                                  #  39    0xfbe62  9      OPC=movl_m32_imm32  
  addl %ebx, %eax                                                              #  40    0xfbe6b  2      OPC=addl_r32_r32    
  movl %eax, %eax                                                              #  41    0xfbe6d  2      OPC=movl_r32_r32    
  movl %edx, (%r15,%rax,1)                                                     #  42    0xfbe6f  4      OPC=movl_m32_r32    
  movl %ebx, %ebx                                                              #  43    0xfbe73  2      OPC=movl_r32_r32    
  movl (%r15,%rbx,1), %eax                                                     #  44    0xfbe75  4      OPC=movl_r32_m32    
  subl $0xc, %eax                                                              #  45    0xfbe79  3      OPC=subl_r32_imm8   
  nop                                                                          #  46    0xfbe7c  1      OPC=nop             
  nop                                                                          #  47    0xfbe7d  1      OPC=nop             
  nop                                                                          #  48    0xfbe7e  1      OPC=nop             
  nop                                                                          #  49    0xfbe7f  1      OPC=nop             
  movl %eax, %eax                                                              #  50    0xfbe80  2      OPC=movl_r32_r32    
  movl (%r15,%rax,1), %edi                                                     #  51    0xfbe82  4      OPC=movl_r32_m32    
  addl %ebx, %edi                                                              #  52    0xfbe86  2      OPC=addl_r32_r32    
  nop                                                                          #  53    0xfbe88  1      OPC=nop             
  nop                                                                          #  54    0xfbe89  1      OPC=nop             
  nop                                                                          #  55    0xfbe8a  1      OPC=nop             
  nop                                                                          #  56    0xfbe8b  1      OPC=nop             
  nop                                                                          #  57    0xfbe8c  1      OPC=nop             
  nop                                                                          #  58    0xfbe8d  1      OPC=nop             
  nop                                                                          #  59    0xfbe8e  1      OPC=nop             
  nop                                                                          #  60    0xfbe8f  1      OPC=nop             
  nop                                                                          #  61    0xfbe90  1      OPC=nop             
  nop                                                                          #  62    0xfbe91  1      OPC=nop             
  nop                                                                          #  63    0xfbe92  1      OPC=nop             
  nop                                                                          #  64    0xfbe93  1      OPC=nop             
  nop                                                                          #  65    0xfbe94  1      OPC=nop             
  nop                                                                          #  66    0xfbe95  1      OPC=nop             
  nop                                                                          #  67    0xfbe96  1      OPC=nop             
  nop                                                                          #  68    0xfbe97  1      OPC=nop             
  nop                                                                          #  69    0xfbe98  1      OPC=nop             
  nop                                                                          #  70    0xfbe99  1      OPC=nop             
  nop                                                                          #  71    0xfbe9a  1      OPC=nop             
  callq ._ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E  #  72    0xfbe9b  5      OPC=callq_label     
  leal 0xc(%r12), %eax                                                         #  73    0xfbea0  5      OPC=leal_r32_m16    
  leal 0x8(%rbx), %edi                                                         #  74    0xfbea5  3      OPC=leal_r32_m16    
  movl 0x8(%rsp), %esi                                                         #  75    0xfbea8  4      OPC=movl_r32_m32    
  movl %eax, %eax                                                              #  76    0xfbeac  2      OPC=movl_r32_r32    
  movl (%r15,%rax,1), %edx                                                     #  77    0xfbeae  4      OPC=movl_r32_m32    
  movl %eax, %eax                                                              #  78    0xfbeb2  2      OPC=movl_r32_r32    
  movl 0x4(%r15,%rax,1), %eax                                                  #  79    0xfbeb4  5      OPC=movl_r32_m32    
  movl %ebx, %ebx                                                              #  80    0xfbeb9  2      OPC=movl_r32_r32    
  movl %edx, 0x8(%r15,%rbx,1)                                                  #  81    0xfbebb  5      OPC=movl_m32_r32    
  subl $0xc, %edx                                                              #  82    0xfbec0  3      OPC=subl_r32_imm8   
  movl %edx, %edx                                                              #  83    0xfbec3  2      OPC=movl_r32_r32    
  movl (%r15,%rdx,1), %edx                                                     #  84    0xfbec5  4      OPC=movl_r32_m32    
  addl %edi, %edx                                                              #  85    0xfbec9  2      OPC=addl_r32_r32    
  movl %edx, %edx                                                              #  86    0xfbecb  2      OPC=movl_r32_r32    
  movl %eax, (%r15,%rdx,1)                                                     #  87    0xfbecd  4      OPC=movl_m32_r32    
  movl %ebx, %ebx                                                              #  88    0xfbed1  2      OPC=movl_r32_r32    
  movl 0x8(%r15,%rbx,1), %eax                                                  #  89    0xfbed3  5      OPC=movl_r32_m32    
  subl $0xc, %eax                                                              #  90    0xfbed8  3      OPC=subl_r32_imm8   
  nop                                                                          #  91    0xfbedb  1      OPC=nop             
  nop                                                                          #  92    0xfbedc  1      OPC=nop             
  nop                                                                          #  93    0xfbedd  1      OPC=nop             
  nop                                                                          #  94    0xfbede  1      OPC=nop             
  nop                                                                          #  95    0xfbedf  1      OPC=nop             
  movl %eax, %eax                                                              #  96    0xfbee0  2      OPC=movl_r32_r32    
  addl (%r15,%rax,1), %edi                                                     #  97    0xfbee2  4      OPC=addl_r32_m32    
  nop                                                                          #  98    0xfbee6  1      OPC=nop             
  nop                                                                          #  99    0xfbee7  1      OPC=nop             
  nop                                                                          #  100   0xfbee8  1      OPC=nop             
  nop                                                                          #  101   0xfbee9  1      OPC=nop             
  nop                                                                          #  102   0xfbeea  1      OPC=nop             
  nop                                                                          #  103   0xfbeeb  1      OPC=nop             
  nop                                                                          #  104   0xfbeec  1      OPC=nop             
  nop                                                                          #  105   0xfbeed  1      OPC=nop             
  nop                                                                          #  106   0xfbeee  1      OPC=nop             
  nop                                                                          #  107   0xfbeef  1      OPC=nop             
  nop                                                                          #  108   0xfbef0  1      OPC=nop             
  nop                                                                          #  109   0xfbef1  1      OPC=nop             
  nop                                                                          #  110   0xfbef2  1      OPC=nop             
  nop                                                                          #  111   0xfbef3  1      OPC=nop             
  nop                                                                          #  112   0xfbef4  1      OPC=nop             
  nop                                                                          #  113   0xfbef5  1      OPC=nop             
  nop                                                                          #  114   0xfbef6  1      OPC=nop             
  nop                                                                          #  115   0xfbef7  1      OPC=nop             
  nop                                                                          #  116   0xfbef8  1      OPC=nop             
  nop                                                                          #  117   0xfbef9  1      OPC=nop             
  nop                                                                          #  118   0xfbefa  1      OPC=nop             
  callq ._ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E  #  119   0xfbefb  5      OPC=callq_label     
  movl %r12d, %r12d                                                            #  120   0xfbf00  3      OPC=movl_r32_r32    
  movl (%r15,%r12,1), %eax                                                     #  121   0xfbf03  4      OPC=movl_r32_m32    
  movl %r12d, %r12d                                                            #  122   0xfbf07  3      OPC=movl_r32_r32    
  movl 0x14(%r15,%r12,1), %edx                                                 #  123   0xfbf0a  5      OPC=movl_r32_m32    
  movl %ebx, %ebx                                                              #  124   0xfbf0f  2      OPC=movl_r32_r32    
  movl %eax, (%r15,%rbx,1)                                                     #  125   0xfbf11  4      OPC=movl_m32_r32    
  subl $0xc, %eax                                                              #  126   0xfbf15  3      OPC=subl_r32_imm8   
  movl %eax, %eax                                                              #  127   0xfbf18  2      OPC=movl_r32_r32    
  movl (%r15,%rax,1), %eax                                                     #  128   0xfbf1a  4      OPC=movl_r32_m32    
  addl %ebx, %eax                                                              #  129   0xfbf1e  2      OPC=addl_r32_r32    
  movl %eax, %eax                                                              #  130   0xfbf20  2      OPC=movl_r32_r32    
  movl %edx, (%r15,%rax,1)                                                     #  131   0xfbf22  4      OPC=movl_m32_r32    
  movl %r12d, %r12d                                                            #  132   0xfbf26  3      OPC=movl_r32_r32    
  movl 0x18(%r15,%r12,1), %eax                                                 #  133   0xfbf29  5      OPC=movl_r32_m32    
  movq 0x28(%rsp), %r13                                                        #  134   0xfbf2e  5      OPC=movq_r64_m64    
  movq 0x20(%rsp), %r12                                                        #  135   0xfbf33  5      OPC=movq_r64_m64    
  movq 0x30(%rsp), %r14                                                        #  136   0xfbf38  5      OPC=movq_r64_m64    
  nop                                                                          #  137   0xfbf3d  1      OPC=nop             
  nop                                                                          #  138   0xfbf3e  1      OPC=nop             
  nop                                                                          #  139   0xfbf3f  1      OPC=nop             
  movl %ebx, %ebx                                                              #  140   0xfbf40  2      OPC=movl_r32_r32    
  movl %eax, 0x8(%r15,%rbx,1)                                                  #  141   0xfbf42  5      OPC=movl_m32_r32    
  movq 0x18(%rsp), %rbx                                                        #  142   0xfbf47  5      OPC=movq_r64_m64    
  addl $0x38, %esp                                                             #  143   0xfbf4c  3      OPC=addl_r32_imm8   
  addq %r15, %rsp                                                              #  144   0xfbf4f  3      OPC=addq_r64_r64    
  popq %r11                                                                    #  145   0xfbf52  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d                                                      #  146   0xfbf54  7      OPC=andl_r32_imm32  
  nop                                                                          #  147   0xfbf5b  1      OPC=nop             
  nop                                                                          #  148   0xfbf5c  1      OPC=nop             
  nop                                                                          #  149   0xfbf5d  1      OPC=nop             
  nop                                                                          #  150   0xfbf5e  1      OPC=nop             
  addq %r15, %r11                                                              #  151   0xfbf5f  3      OPC=addq_r64_r64    
  jmpq %r11                                                                    #  152   0xfbf62  3      OPC=jmpq_r64        
  xchgw %ax, %ax                                                               #  153   0xfbf65  2      OPC=xchgw_ax_r16    
.L_fbf60:                                                                      #        0xfbf67  0      OPC=<label>         
  movl %eax, %edi                                                              #  154   0xfbf67  2      OPC=movl_r32_r32    
  nop                                                                          #  155   0xfbf69  1      OPC=nop             
  nop                                                                          #  156   0xfbf6a  1      OPC=nop             
  nop                                                                          #  157   0xfbf6b  1      OPC=nop             
  nop                                                                          #  158   0xfbf6c  1      OPC=nop             
  nop                                                                          #  159   0xfbf6d  1      OPC=nop             
  nop                                                                          #  160   0xfbf6e  1      OPC=nop             
  nop                                                                          #  161   0xfbf6f  1      OPC=nop             
  nop                                                                          #  162   0xfbf70  1      OPC=nop             
  nop                                                                          #  163   0xfbf71  1      OPC=nop             
  nop                                                                          #  164   0xfbf72  1      OPC=nop             
  nop                                                                          #  165   0xfbf73  1      OPC=nop             
  nop                                                                          #  166   0xfbf74  1      OPC=nop             
  nop                                                                          #  167   0xfbf75  1      OPC=nop             
  nop                                                                          #  168   0xfbf76  1      OPC=nop             
  nop                                                                          #  169   0xfbf77  1      OPC=nop             
  nop                                                                          #  170   0xfbf78  1      OPC=nop             
  nop                                                                          #  171   0xfbf79  1      OPC=nop             
  nop                                                                          #  172   0xfbf7a  1      OPC=nop             
  nop                                                                          #  173   0xfbf7b  1      OPC=nop             
  nop                                                                          #  174   0xfbf7c  1      OPC=nop             
  nop                                                                          #  175   0xfbf7d  1      OPC=nop             
  nop                                                                          #  176   0xfbf7e  1      OPC=nop             
  nop                                                                          #  177   0xfbf7f  1      OPC=nop             
  nop                                                                          #  178   0xfbf80  1      OPC=nop             
  nop                                                                          #  179   0xfbf81  1      OPC=nop             
  callq ._Unwind_Resume                                                        #  180   0xfbf82  5      OPC=callq_label     
  movl %r14d, %r14d                                                            #  181   0xfbf87  3      OPC=movl_r32_r32    
  movl (%r15,%r14,1), %edx                                                     #  182   0xfbf8a  4      OPC=movl_r32_m32    
  movl %r13d, %r13d                                                            #  183   0xfbf8e  3      OPC=movl_r32_r32    
  movl (%r15,%r13,1), %ecx                                                     #  184   0xfbf91  4      OPC=movl_r32_m32    
  movl %ebx, %ebx                                                              #  185   0xfbf95  2      OPC=movl_r32_r32    
  movl %edx, (%r15,%rbx,1)                                                     #  186   0xfbf97  4      OPC=movl_m32_r32    
  subl $0xc, %edx                                                              #  187   0xfbf9b  3      OPC=subl_r32_imm8   
  movl %edx, %edx                                                              #  188   0xfbf9e  2      OPC=movl_r32_r32    
  movl (%r15,%rdx,1), %edx                                                     #  189   0xfbfa0  4      OPC=movl_r32_m32    
  nop                                                                          #  190   0xfbfa4  1      OPC=nop             
  nop                                                                          #  191   0xfbfa5  1      OPC=nop             
  nop                                                                          #  192   0xfbfa6  1      OPC=nop             
  movl %ebx, %ebx                                                              #  193   0xfbfa7  2      OPC=movl_r32_r32    
  movl $0x0, 0x4(%r15,%rbx,1)                                                  #  194   0xfbfa9  9      OPC=movl_m32_imm32  
  addl %ebx, %edx                                                              #  195   0xfbfb2  2      OPC=addl_r32_r32    
  movl %edx, %edx                                                              #  196   0xfbfb4  2      OPC=movl_r32_r32    
  movl %ecx, (%r15,%rdx,1)                                                     #  197   0xfbfb6  4      OPC=movl_m32_r32    
  jmpq .L_fbf60                                                                #  198   0xfbfba  2      OPC=jmpq_label      
  nop                                                                          #  199   0xfbfbc  1      OPC=nop             
  nop                                                                          #  200   0xfbfbd  1      OPC=nop             
  nop                                                                          #  201   0xfbfbe  1      OPC=nop             
  nop                                                                          #  202   0xfbfbf  1      OPC=nop             
  nop                                                                          #  203   0xfbfc0  1      OPC=nop             
  nop                                                                          #  204   0xfbfc1  1      OPC=nop             
  nop                                                                          #  205   0xfbfc2  1      OPC=nop             
  nop                                                                          #  206   0xfbfc3  1      OPC=nop             
  nop                                                                          #  207   0xfbfc4  1      OPC=nop             
  nop                                                                          #  208   0xfbfc5  1      OPC=nop             
  nop                                                                          #  209   0xfbfc6  1      OPC=nop             
                                                                                                                            
.size _ZNSdC2EPSt15basic_streambufIcSt11char_traitsIcEE, .-_ZNSdC2EPSt15basic_streambufIcSt11char_traitsIcEE

