  .text
  .globl _ZNKSt7codecvtIwc10_mbstate_tE5do_inERS0_PKcS4_RS4_PwS6_RS6_
  .type _ZNKSt7codecvtIwc10_mbstate_tE5do_inERS0_PKcS4_RS4_PwS6_RS6_, @function

#! file-offset 0x11abc0
#! rip-offset  0xdabc0
#! capacity    416 bytes

# Text                                                          #  Line  RIP      Bytes  Opcode              
._ZNKSt7codecvtIwc10_mbstate_tE5do_inERS0_PKcS4_RS4_PwS6_RS6_:  #        0xdabc0  0      OPC=<label>         
  pushq %r14                                                    #  1     0xdabc0  2      OPC=pushq_r64_1     
  movl %ecx, %ecx                                               #  2     0xdabc2  2      OPC=movl_r32_r32    
  movl %r8d, %r8d                                               #  3     0xdabc4  3      OPC=movl_r32_r32    
  pushq %r13                                                    #  4     0xdabc7  2      OPC=pushq_r64_1     
  movl %r9d, %r13d                                              #  5     0xdabc9  3      OPC=movl_r32_r32    
  pushq %r12                                                    #  6     0xdabcc  2      OPC=pushq_r64_1     
  movl %esi, %r12d                                              #  7     0xdabce  3      OPC=movl_r32_r32    
  pushq %rbx                                                    #  8     0xdabd1  1      OPC=pushq_r64_1     
  movl %edx, %ebx                                               #  9     0xdabd2  2      OPC=movl_r32_r32    
  subl $0x48, %esp                                              #  10    0xdabd4  3      OPC=subl_r32_imm8   
  addq %r15, %rsp                                               #  11    0xdabd7  3      OPC=addq_r64_r64    
  movl 0x70(%rsp), %eax                                         #  12    0xdabda  4      OPC=movl_r32_m32    
  xchgw %ax, %ax                                                #  13    0xdabde  2      OPC=xchgw_ax_r16    
  movl 0x78(%rsp), %edx                                         #  14    0xdabe0  4      OPC=movl_r32_m32    
  movq %rcx, 0x10(%rsp)                                         #  15    0xdabe4  5      OPC=movq_m64_r64    
  cmpl 0x10(%rsp), %ebx                                         #  16    0xdabe9  4      OPC=cmpl_r32_m32    
  movq %r8, 0x20(%rsp)                                          #  17    0xdabed  5      OPC=movq_m64_r64    
  movq %rax, 0x18(%rsp)                                         #  18    0xdabf2  5      OPC=movq_m64_r64    
  movq %rdx, 0x28(%rsp)                                         #  19    0xdabf7  5      OPC=movq_m64_r64    
  setb %dl                                                      #  20    0xdabfc  3      OPC=setb_r8         
  nop                                                           #  21    0xdabff  1      OPC=nop             
  cmpl 0x18(%rsp), %r13d                                        #  22    0xdac00  5      OPC=cmpl_r32_m32    
  movl %r12d, %r12d                                             #  23    0xdac05  3      OPC=movl_r32_r32    
  movq (%r15,%r12,1), %rax                                      #  24    0xdac08  4      OPC=movq_r64_m64    
  movq %rax, 0x30(%rsp)                                         #  25    0xdac0c  5      OPC=movq_m64_r64    
  movl %edx, %eax                                               #  26    0xdac11  2      OPC=movl_r32_r32    
  jb .L_dac80                                                   #  27    0xdac13  2      OPC=jb_label        
  nop                                                           #  28    0xdac15  1      OPC=nop             
  nop                                                           #  29    0xdac16  1      OPC=nop             
  nop                                                           #  30    0xdac17  1      OPC=nop             
  nop                                                           #  31    0xdac18  1      OPC=nop             
  nop                                                           #  32    0xdac19  1      OPC=nop             
  nop                                                           #  33    0xdac1a  1      OPC=nop             
  nop                                                           #  34    0xdac1b  1      OPC=nop             
  nop                                                           #  35    0xdac1c  1      OPC=nop             
  nop                                                           #  36    0xdac1d  1      OPC=nop             
  nop                                                           #  37    0xdac1e  1      OPC=nop             
  nop                                                           #  38    0xdac1f  1      OPC=nop             
.L_dac20:                                                       #        0xdac20  0      OPC=<label>         
  andl $0x1, %eax                                               #  39    0xdac20  3      OPC=andl_r32_imm8   
  nop                                                           #  40    0xdac23  1      OPC=nop             
  nop                                                           #  41    0xdac24  1      OPC=nop             
  nop                                                           #  42    0xdac25  1      OPC=nop             
  nop                                                           #  43    0xdac26  1      OPC=nop             
  nop                                                           #  44    0xdac27  1      OPC=nop             
  nop                                                           #  45    0xdac28  1      OPC=nop             
  nop                                                           #  46    0xdac29  1      OPC=nop             
  nop                                                           #  47    0xdac2a  1      OPC=nop             
  nop                                                           #  48    0xdac2b  1      OPC=nop             
  nop                                                           #  49    0xdac2c  1      OPC=nop             
  nop                                                           #  50    0xdac2d  1      OPC=nop             
  nop                                                           #  51    0xdac2e  1      OPC=nop             
  nop                                                           #  52    0xdac2f  1      OPC=nop             
  nop                                                           #  53    0xdac30  1      OPC=nop             
  nop                                                           #  54    0xdac31  1      OPC=nop             
  nop                                                           #  55    0xdac32  1      OPC=nop             
  nop                                                           #  56    0xdac33  1      OPC=nop             
  nop                                                           #  57    0xdac34  1      OPC=nop             
  nop                                                           #  58    0xdac35  1      OPC=nop             
  nop                                                           #  59    0xdac36  1      OPC=nop             
  nop                                                           #  60    0xdac37  1      OPC=nop             
  nop                                                           #  61    0xdac38  1      OPC=nop             
  nop                                                           #  62    0xdac39  1      OPC=nop             
  nop                                                           #  63    0xdac3a  1      OPC=nop             
  nop                                                           #  64    0xdac3b  1      OPC=nop             
  nop                                                           #  65    0xdac3c  1      OPC=nop             
  nop                                                           #  66    0xdac3d  1      OPC=nop             
  nop                                                           #  67    0xdac3e  1      OPC=nop             
  nop                                                           #  68    0xdac3f  1      OPC=nop             
.L_dac40:                                                       #        0xdac40  0      OPC=<label>         
  movq 0x20(%rsp), %rdx                                         #  69    0xdac40  5      OPC=movq_r64_m64    
  movl %edx, %edx                                               #  70    0xdac45  2      OPC=movl_r32_r32    
  movl %ebx, (%r15,%rdx,1)                                      #  71    0xdac47  4      OPC=movl_m32_r32    
  movq 0x28(%rsp), %rdx                                         #  72    0xdac4b  5      OPC=movq_r64_m64    
  movl %edx, %edx                                               #  73    0xdac50  2      OPC=movl_r32_r32    
  movl %r13d, (%r15,%rdx,1)                                     #  74    0xdac52  4      OPC=movl_m32_r32    
  addl $0x48, %esp                                              #  75    0xdac56  3      OPC=addl_r32_imm8   
  addq %r15, %rsp                                               #  76    0xdac59  3      OPC=addq_r64_r64    
  popq %rbx                                                     #  77    0xdac5c  1      OPC=popq_r64_1      
  popq %r12                                                     #  78    0xdac5d  2      OPC=popq_r64_1      
  nop                                                           #  79    0xdac5f  1      OPC=nop             
  popq %r13                                                     #  80    0xdac60  2      OPC=popq_r64_1      
  popq %r14                                                     #  81    0xdac62  2      OPC=popq_r64_1      
  popq %r11                                                     #  82    0xdac64  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d                                       #  83    0xdac66  7      OPC=andl_r32_imm32  
  nop                                                           #  84    0xdac6d  1      OPC=nop             
  nop                                                           #  85    0xdac6e  1      OPC=nop             
  nop                                                           #  86    0xdac6f  1      OPC=nop             
  nop                                                           #  87    0xdac70  1      OPC=nop             
  addq %r15, %r11                                               #  88    0xdac71  3      OPC=addq_r64_r64    
  jmpq %r11                                                     #  89    0xdac74  3      OPC=jmpq_r64        
  nop                                                           #  90    0xdac77  1      OPC=nop             
  nop                                                           #  91    0xdac78  1      OPC=nop             
  nop                                                           #  92    0xdac79  1      OPC=nop             
  nop                                                           #  93    0xdac7a  1      OPC=nop             
  nop                                                           #  94    0xdac7b  1      OPC=nop             
  nop                                                           #  95    0xdac7c  1      OPC=nop             
  nop                                                           #  96    0xdac7d  1      OPC=nop             
  nop                                                           #  97    0xdac7e  1      OPC=nop             
  nop                                                           #  98    0xdac7f  1      OPC=nop             
  nop                                                           #  99    0xdac80  1      OPC=nop             
  nop                                                           #  100   0xdac81  1      OPC=nop             
  nop                                                           #  101   0xdac82  1      OPC=nop             
  nop                                                           #  102   0xdac83  1      OPC=nop             
  nop                                                           #  103   0xdac84  1      OPC=nop             
  nop                                                           #  104   0xdac85  1      OPC=nop             
  nop                                                           #  105   0xdac86  1      OPC=nop             
.L_dac80:                                                       #        0xdac87  0      OPC=<label>         
  testb %dl, %dl                                                #  106   0xdac87  2      OPC=testb_r8_r8     
  je .L_dac20                                                   #  107   0xdac89  2      OPC=je_label        
  leal 0x30(%rsp), %eax                                         #  108   0xdac8b  4      OPC=leal_r32_m16    
  movl 0x10(%rsp), %r14d                                        #  109   0xdac8f  5      OPC=movl_r32_m32    
  movq %rax, 0x8(%rsp)                                          #  110   0xdac94  5      OPC=movq_m64_r64    
  nop                                                           #  111   0xdac99  1      OPC=nop             
  nop                                                           #  112   0xdac9a  1      OPC=nop             
  nop                                                           #  113   0xdac9b  1      OPC=nop             
  nop                                                           #  114   0xdac9c  1      OPC=nop             
  nop                                                           #  115   0xdac9d  1      OPC=nop             
  nop                                                           #  116   0xdac9e  1      OPC=nop             
  nop                                                           #  117   0xdac9f  1      OPC=nop             
  nop                                                           #  118   0xdaca0  1      OPC=nop             
  nop                                                           #  119   0xdaca1  1      OPC=nop             
  nop                                                           #  120   0xdaca2  1      OPC=nop             
  nop                                                           #  121   0xdaca3  1      OPC=nop             
  nop                                                           #  122   0xdaca4  1      OPC=nop             
  nop                                                           #  123   0xdaca5  1      OPC=nop             
  nop                                                           #  124   0xdaca6  1      OPC=nop             
.L_daca0:                                                       #        0xdaca7  0      OPC=<label>         
  movl 0x8(%rsp), %ecx                                          #  125   0xdaca7  4      OPC=movl_r32_m32    
  movl %r14d, %edx                                              #  126   0xdacab  3      OPC=movl_r32_r32    
  movl %ebx, %esi                                               #  127   0xdacae  2      OPC=movl_r32_r32    
  subl %ebx, %edx                                               #  128   0xdacb0  2      OPC=subl_r32_r32    
  movl %r13d, %edi                                              #  129   0xdacb2  3      OPC=movl_r32_r32    
  nop                                                           #  130   0xdacb5  1      OPC=nop             
  nop                                                           #  131   0xdacb6  1      OPC=nop             
  nop                                                           #  132   0xdacb7  1      OPC=nop             
  nop                                                           #  133   0xdacb8  1      OPC=nop             
  nop                                                           #  134   0xdacb9  1      OPC=nop             
  nop                                                           #  135   0xdacba  1      OPC=nop             
  nop                                                           #  136   0xdacbb  1      OPC=nop             
  nop                                                           #  137   0xdacbc  1      OPC=nop             
  nop                                                           #  138   0xdacbd  1      OPC=nop             
  nop                                                           #  139   0xdacbe  1      OPC=nop             
  nop                                                           #  140   0xdacbf  1      OPC=nop             
  nop                                                           #  141   0xdacc0  1      OPC=nop             
  nop                                                           #  142   0xdacc1  1      OPC=nop             
  callq .mbrtowc                                                #  143   0xdacc2  5      OPC=callq_label     
  cmpl $0xffffffff, %eax                                        #  144   0xdacc7  6      OPC=cmpl_r32_imm32  
  nop                                                           #  145   0xdaccd  1      OPC=nop             
  nop                                                           #  146   0xdacce  1      OPC=nop             
  nop                                                           #  147   0xdaccf  1      OPC=nop             
  je .L_dad20                                                   #  148   0xdacd0  2      OPC=je_label        
  cmpl $0xfffffffe, %eax                                        #  149   0xdacd2  6      OPC=cmpl_r32_imm32  
  nop                                                           #  150   0xdacd8  1      OPC=nop             
  nop                                                           #  151   0xdacd9  1      OPC=nop             
  nop                                                           #  152   0xdacda  1      OPC=nop             
  je .L_dad40                                                   #  153   0xdacdb  2      OPC=je_label        
  testl %eax, %eax                                              #  154   0xdacdd  2      OPC=testl_r32_r32   
  jne .L_dace0                                                  #  155   0xdacdf  2      OPC=jne_label       
  movl %r13d, %r13d                                             #  156   0xdace1  3      OPC=movl_r32_r32    
  movl $0x0, (%r15,%r13,1)                                      #  157   0xdace4  8      OPC=movl_m32_imm32  
  movb $0x1, %al                                                #  158   0xdacec  2      OPC=movb_r8_imm8    
  nop                                                           #  159   0xdacee  1      OPC=nop             
  nop                                                           #  160   0xdacef  1      OPC=nop             
  nop                                                           #  161   0xdacf0  1      OPC=nop             
  nop                                                           #  162   0xdacf1  1      OPC=nop             
  nop                                                           #  163   0xdacf2  1      OPC=nop             
.L_dace0:                                                       #        0xdacf3  0      OPC=<label>         
  leal (%rax,%rbx,1), %ebx                                      #  164   0xdacf3  3      OPC=leal_r32_m16    
  addl $0x4, %r13d                                              #  165   0xdacf6  4      OPC=addl_r32_imm8   
  cmpl %ebx, 0x10(%rsp)                                         #  166   0xdacfa  4      OPC=cmpl_m32_r32    
  movq 0x30(%rsp), %rdx                                         #  167   0xdacfe  5      OPC=movq_r64_m64    
  seta %al                                                      #  168   0xdad03  3      OPC=seta_r8         
  movl %r12d, %r12d                                             #  169   0xdad06  3      OPC=movl_r32_r32    
  movq %rdx, (%r15,%r12,1)                                      #  170   0xdad09  4      OPC=movq_m64_r64    
  jbe .L_dac20                                                  #  171   0xdad0d  6      OPC=jbe_label_1     
  cmpl %r13d, 0x18(%rsp)                                        #  172   0xdad13  5      OPC=cmpl_m32_r32    
  ja .L_daca0                                                   #  173   0xdad18  2      OPC=ja_label        
  jmpq .L_dac20                                                 #  174   0xdad1a  5      OPC=jmpq_label_1    
  nop                                                           #  175   0xdad1f  1      OPC=nop             
  nop                                                           #  176   0xdad20  1      OPC=nop             
  nop                                                           #  177   0xdad21  1      OPC=nop             
  nop                                                           #  178   0xdad22  1      OPC=nop             
  nop                                                           #  179   0xdad23  1      OPC=nop             
  nop                                                           #  180   0xdad24  1      OPC=nop             
  nop                                                           #  181   0xdad25  1      OPC=nop             
  nop                                                           #  182   0xdad26  1      OPC=nop             
  nop                                                           #  183   0xdad27  1      OPC=nop             
  nop                                                           #  184   0xdad28  1      OPC=nop             
  nop                                                           #  185   0xdad29  1      OPC=nop             
  nop                                                           #  186   0xdad2a  1      OPC=nop             
  nop                                                           #  187   0xdad2b  1      OPC=nop             
  nop                                                           #  188   0xdad2c  1      OPC=nop             
  nop                                                           #  189   0xdad2d  1      OPC=nop             
  nop                                                           #  190   0xdad2e  1      OPC=nop             
  nop                                                           #  191   0xdad2f  1      OPC=nop             
  nop                                                           #  192   0xdad30  1      OPC=nop             
  nop                                                           #  193   0xdad31  1      OPC=nop             
  nop                                                           #  194   0xdad32  1      OPC=nop             
.L_dad20:                                                       #        0xdad33  0      OPC=<label>         
  movl $0x2, %eax                                               #  195   0xdad33  5      OPC=movl_r32_imm32  
  jmpq .L_dac40                                                 #  196   0xdad38  5      OPC=jmpq_label_1    
  nop                                                           #  197   0xdad3d  1      OPC=nop             
  nop                                                           #  198   0xdad3e  1      OPC=nop             
  nop                                                           #  199   0xdad3f  1      OPC=nop             
  nop                                                           #  200   0xdad40  1      OPC=nop             
  nop                                                           #  201   0xdad41  1      OPC=nop             
  nop                                                           #  202   0xdad42  1      OPC=nop             
  nop                                                           #  203   0xdad43  1      OPC=nop             
  nop                                                           #  204   0xdad44  1      OPC=nop             
  nop                                                           #  205   0xdad45  1      OPC=nop             
  nop                                                           #  206   0xdad46  1      OPC=nop             
  nop                                                           #  207   0xdad47  1      OPC=nop             
  nop                                                           #  208   0xdad48  1      OPC=nop             
  nop                                                           #  209   0xdad49  1      OPC=nop             
  nop                                                           #  210   0xdad4a  1      OPC=nop             
  nop                                                           #  211   0xdad4b  1      OPC=nop             
  nop                                                           #  212   0xdad4c  1      OPC=nop             
  nop                                                           #  213   0xdad4d  1      OPC=nop             
  nop                                                           #  214   0xdad4e  1      OPC=nop             
  nop                                                           #  215   0xdad4f  1      OPC=nop             
  nop                                                           #  216   0xdad50  1      OPC=nop             
  nop                                                           #  217   0xdad51  1      OPC=nop             
  nop                                                           #  218   0xdad52  1      OPC=nop             
.L_dad40:                                                       #        0xdad53  0      OPC=<label>         
  movl $0x1, %eax                                               #  219   0xdad53  5      OPC=movl_r32_imm32  
  jmpq .L_dac40                                                 #  220   0xdad58  5      OPC=jmpq_label_1    
  nop                                                           #  221   0xdad5d  1      OPC=nop             
  nop                                                           #  222   0xdad5e  1      OPC=nop             
  nop                                                           #  223   0xdad5f  1      OPC=nop             
  nop                                                           #  224   0xdad60  1      OPC=nop             
  nop                                                           #  225   0xdad61  1      OPC=nop             
  nop                                                           #  226   0xdad62  1      OPC=nop             
  nop                                                           #  227   0xdad63  1      OPC=nop             
  nop                                                           #  228   0xdad64  1      OPC=nop             
  nop                                                           #  229   0xdad65  1      OPC=nop             
  nop                                                           #  230   0xdad66  1      OPC=nop             
  nop                                                           #  231   0xdad67  1      OPC=nop             
  nop                                                           #  232   0xdad68  1      OPC=nop             
  nop                                                           #  233   0xdad69  1      OPC=nop             
  nop                                                           #  234   0xdad6a  1      OPC=nop             
  nop                                                           #  235   0xdad6b  1      OPC=nop             
  nop                                                           #  236   0xdad6c  1      OPC=nop             
  nop                                                           #  237   0xdad6d  1      OPC=nop             
  nop                                                           #  238   0xdad6e  1      OPC=nop             
  nop                                                           #  239   0xdad6f  1      OPC=nop             
  nop                                                           #  240   0xdad70  1      OPC=nop             
  nop                                                           #  241   0xdad71  1      OPC=nop             
  nop                                                           #  242   0xdad72  1      OPC=nop             
                                                                                                             
.size _ZNKSt7codecvtIwc10_mbstate_tE5do_inERS0_PKcS4_RS4_PwS6_RS6_, .-_ZNKSt7codecvtIwc10_mbstate_tE5do_inERS0_PKcS4_RS4_PwS6_RS6_

