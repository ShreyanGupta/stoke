  .text
  .globl _ZNSs6assignERKSs
  .type _ZNSs6assignERKSs, @function

#! file-offset 0xed020
#! rip-offset  0xad020
#! capacity    416 bytes

# Text                                   #  Line  RIP      Bytes  Opcode              
._ZNSs6assignERKSs:                      #        0xad020  0      OPC=<label>         
  pushq %rbx                             #  1     0xad020  1      OPC=pushq_r64_1     
  movl %esi, %esi                        #  2     0xad021  2      OPC=movl_r32_r32    
  movl %edi, %ebx                        #  3     0xad023  2      OPC=movl_r32_r32    
  subl $0x20, %esp                       #  4     0xad025  3      OPC=subl_r32_imm8   
  addq %r15, %rsp                        #  5     0xad028  3      OPC=addq_r64_r64    
  movl %ebx, %ebx                        #  6     0xad02b  2      OPC=movl_r32_r32    
  movl (%r15,%rbx,1), %edx               #  7     0xad02d  4      OPC=movl_r32_m32    
  movl %esi, %esi                        #  8     0xad031  2      OPC=movl_r32_r32    
  movl (%r15,%rsi,1), %edi               #  9     0xad033  4      OPC=movl_r32_m32    
  cmpl %edi, %edx                        #  10    0xad037  2      OPC=cmpl_r32_r32    
  je .L_ad0c0                            #  11    0xad039  6      OPC=je_label_1      
  nop                                    #  12    0xad03f  1      OPC=nop             
  subl $0xc, %edi                        #  13    0xad040  3      OPC=subl_r32_imm8   
  movl %edi, %edi                        #  14    0xad043  2      OPC=movl_r32_r32    
  movl 0x8(%r15,%rdi,1), %eax            #  15    0xad045  5      OPC=movl_r32_m32    
  testl %eax, %eax                       #  16    0xad04a  2      OPC=testl_r32_r32   
  js .L_ad0e0                            #  17    0xad04c  6      OPC=js_label_1      
  movl $0x10073580, %ecx                 #  18    0xad052  5      OPC=movl_r32_imm32  
  cmpl %ecx, %edi                        #  19    0xad057  2      OPC=cmpl_r32_r32    
  jne .L_ad120                           #  20    0xad059  6      OPC=jne_label_1     
  nop                                    #  21    0xad05f  1      OPC=nop             
.L_ad060:                                #        0xad060  0      OPC=<label>         
  leal 0xc(%rdi), %eax                   #  22    0xad060  3      OPC=leal_r32_m16    
  nop                                    #  23    0xad063  1      OPC=nop             
  nop                                    #  24    0xad064  1      OPC=nop             
  nop                                    #  25    0xad065  1      OPC=nop             
  nop                                    #  26    0xad066  1      OPC=nop             
  nop                                    #  27    0xad067  1      OPC=nop             
  nop                                    #  28    0xad068  1      OPC=nop             
  nop                                    #  29    0xad069  1      OPC=nop             
  nop                                    #  30    0xad06a  1      OPC=nop             
  nop                                    #  31    0xad06b  1      OPC=nop             
  nop                                    #  32    0xad06c  1      OPC=nop             
  nop                                    #  33    0xad06d  1      OPC=nop             
  nop                                    #  34    0xad06e  1      OPC=nop             
  nop                                    #  35    0xad06f  1      OPC=nop             
  nop                                    #  36    0xad070  1      OPC=nop             
  nop                                    #  37    0xad071  1      OPC=nop             
  nop                                    #  38    0xad072  1      OPC=nop             
  nop                                    #  39    0xad073  1      OPC=nop             
  nop                                    #  40    0xad074  1      OPC=nop             
  nop                                    #  41    0xad075  1      OPC=nop             
  nop                                    #  42    0xad076  1      OPC=nop             
  nop                                    #  43    0xad077  1      OPC=nop             
  nop                                    #  44    0xad078  1      OPC=nop             
  nop                                    #  45    0xad079  1      OPC=nop             
  nop                                    #  46    0xad07a  1      OPC=nop             
  nop                                    #  47    0xad07b  1      OPC=nop             
  nop                                    #  48    0xad07c  1      OPC=nop             
  nop                                    #  49    0xad07d  1      OPC=nop             
  nop                                    #  50    0xad07e  1      OPC=nop             
  nop                                    #  51    0xad07f  1      OPC=nop             
.L_ad080:                                #        0xad080  0      OPC=<label>         
  leal -0xc(%rdx), %edi                  #  52    0xad080  3      OPC=leal_r32_m16    
  cmpl %ecx, %edi                        #  53    0xad083  2      OPC=cmpl_r32_r32    
  jne .L_ad140                           #  54    0xad085  6      OPC=jne_label_1     
  nop                                    #  55    0xad08b  1      OPC=nop             
  nop                                    #  56    0xad08c  1      OPC=nop             
  nop                                    #  57    0xad08d  1      OPC=nop             
  nop                                    #  58    0xad08e  1      OPC=nop             
  nop                                    #  59    0xad08f  1      OPC=nop             
  nop                                    #  60    0xad090  1      OPC=nop             
  nop                                    #  61    0xad091  1      OPC=nop             
  nop                                    #  62    0xad092  1      OPC=nop             
  nop                                    #  63    0xad093  1      OPC=nop             
  nop                                    #  64    0xad094  1      OPC=nop             
  nop                                    #  65    0xad095  1      OPC=nop             
  nop                                    #  66    0xad096  1      OPC=nop             
  nop                                    #  67    0xad097  1      OPC=nop             
  nop                                    #  68    0xad098  1      OPC=nop             
  nop                                    #  69    0xad099  1      OPC=nop             
  nop                                    #  70    0xad09a  1      OPC=nop             
  nop                                    #  71    0xad09b  1      OPC=nop             
  nop                                    #  72    0xad09c  1      OPC=nop             
  nop                                    #  73    0xad09d  1      OPC=nop             
  nop                                    #  74    0xad09e  1      OPC=nop             
  nop                                    #  75    0xad09f  1      OPC=nop             
.L_ad0a0:                                #        0xad0a0  0      OPC=<label>         
  movl %ebx, %ebx                        #  76    0xad0a0  2      OPC=movl_r32_r32    
  movl %eax, (%r15,%rbx,1)               #  77    0xad0a2  4      OPC=movl_m32_r32    
  nop                                    #  78    0xad0a6  1      OPC=nop             
  nop                                    #  79    0xad0a7  1      OPC=nop             
  nop                                    #  80    0xad0a8  1      OPC=nop             
  nop                                    #  81    0xad0a9  1      OPC=nop             
  nop                                    #  82    0xad0aa  1      OPC=nop             
  nop                                    #  83    0xad0ab  1      OPC=nop             
  nop                                    #  84    0xad0ac  1      OPC=nop             
  nop                                    #  85    0xad0ad  1      OPC=nop             
  nop                                    #  86    0xad0ae  1      OPC=nop             
  nop                                    #  87    0xad0af  1      OPC=nop             
  nop                                    #  88    0xad0b0  1      OPC=nop             
  nop                                    #  89    0xad0b1  1      OPC=nop             
  nop                                    #  90    0xad0b2  1      OPC=nop             
  nop                                    #  91    0xad0b3  1      OPC=nop             
  nop                                    #  92    0xad0b4  1      OPC=nop             
  nop                                    #  93    0xad0b5  1      OPC=nop             
  nop                                    #  94    0xad0b6  1      OPC=nop             
  nop                                    #  95    0xad0b7  1      OPC=nop             
  nop                                    #  96    0xad0b8  1      OPC=nop             
  nop                                    #  97    0xad0b9  1      OPC=nop             
  nop                                    #  98    0xad0ba  1      OPC=nop             
  nop                                    #  99    0xad0bb  1      OPC=nop             
  nop                                    #  100   0xad0bc  1      OPC=nop             
  nop                                    #  101   0xad0bd  1      OPC=nop             
  nop                                    #  102   0xad0be  1      OPC=nop             
  nop                                    #  103   0xad0bf  1      OPC=nop             
.L_ad0c0:                                #        0xad0c0  0      OPC=<label>         
  movl %ebx, %eax                        #  104   0xad0c0  2      OPC=movl_r32_r32    
  addl $0x20, %esp                       #  105   0xad0c2  3      OPC=addl_r32_imm8   
  addq %r15, %rsp                        #  106   0xad0c5  3      OPC=addq_r64_r64    
  popq %rbx                              #  107   0xad0c8  1      OPC=popq_r64_1      
  popq %r11                              #  108   0xad0c9  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d                #  109   0xad0cb  7      OPC=andl_r32_imm32  
  nop                                    #  110   0xad0d2  1      OPC=nop             
  nop                                    #  111   0xad0d3  1      OPC=nop             
  nop                                    #  112   0xad0d4  1      OPC=nop             
  nop                                    #  113   0xad0d5  1      OPC=nop             
  addq %r15, %r11                        #  114   0xad0d6  3      OPC=addq_r64_r64    
  jmpq %r11                              #  115   0xad0d9  3      OPC=jmpq_r64        
  nop                                    #  116   0xad0dc  1      OPC=nop             
  nop                                    #  117   0xad0dd  1      OPC=nop             
  nop                                    #  118   0xad0de  1      OPC=nop             
  nop                                    #  119   0xad0df  1      OPC=nop             
  nop                                    #  120   0xad0e0  1      OPC=nop             
  nop                                    #  121   0xad0e1  1      OPC=nop             
  nop                                    #  122   0xad0e2  1      OPC=nop             
  nop                                    #  123   0xad0e3  1      OPC=nop             
  nop                                    #  124   0xad0e4  1      OPC=nop             
  nop                                    #  125   0xad0e5  1      OPC=nop             
  nop                                    #  126   0xad0e6  1      OPC=nop             
.L_ad0e0:                                #        0xad0e7  0      OPC=<label>         
  leal 0x1f(%rsp), %esi                  #  127   0xad0e7  4      OPC=leal_r32_m16    
  xorl %edx, %edx                        #  128   0xad0eb  2      OPC=xorl_r32_r32    
  nop                                    #  129   0xad0ed  1      OPC=nop             
  nop                                    #  130   0xad0ee  1      OPC=nop             
  nop                                    #  131   0xad0ef  1      OPC=nop             
  nop                                    #  132   0xad0f0  1      OPC=nop             
  nop                                    #  133   0xad0f1  1      OPC=nop             
  nop                                    #  134   0xad0f2  1      OPC=nop             
  nop                                    #  135   0xad0f3  1      OPC=nop             
  nop                                    #  136   0xad0f4  1      OPC=nop             
  nop                                    #  137   0xad0f5  1      OPC=nop             
  nop                                    #  138   0xad0f6  1      OPC=nop             
  nop                                    #  139   0xad0f7  1      OPC=nop             
  nop                                    #  140   0xad0f8  1      OPC=nop             
  nop                                    #  141   0xad0f9  1      OPC=nop             
  nop                                    #  142   0xad0fa  1      OPC=nop             
  nop                                    #  143   0xad0fb  1      OPC=nop             
  nop                                    #  144   0xad0fc  1      OPC=nop             
  nop                                    #  145   0xad0fd  1      OPC=nop             
  nop                                    #  146   0xad0fe  1      OPC=nop             
  nop                                    #  147   0xad0ff  1      OPC=nop             
  nop                                    #  148   0xad100  1      OPC=nop             
  nop                                    #  149   0xad101  1      OPC=nop             
  callq ._ZNSs4_Rep8_M_cloneERKSaIcEj    #  150   0xad102  5      OPC=callq_label     
  movl %eax, %eax                        #  151   0xad107  2      OPC=movl_r32_r32    
  movl $0x10073580, %ecx                 #  152   0xad109  5      OPC=movl_r32_imm32  
  movl %ebx, %ebx                        #  153   0xad10e  2      OPC=movl_r32_r32    
  movl (%r15,%rbx,1), %edx               #  154   0xad110  4      OPC=movl_r32_m32    
  jmpq .L_ad080                          #  155   0xad114  5      OPC=jmpq_label_1    
  nop                                    #  156   0xad119  1      OPC=nop             
  nop                                    #  157   0xad11a  1      OPC=nop             
  nop                                    #  158   0xad11b  1      OPC=nop             
  nop                                    #  159   0xad11c  1      OPC=nop             
  nop                                    #  160   0xad11d  1      OPC=nop             
  nop                                    #  161   0xad11e  1      OPC=nop             
  nop                                    #  162   0xad11f  1      OPC=nop             
  nop                                    #  163   0xad120  1      OPC=nop             
  nop                                    #  164   0xad121  1      OPC=nop             
  nop                                    #  165   0xad122  1      OPC=nop             
  nop                                    #  166   0xad123  1      OPC=nop             
  nop                                    #  167   0xad124  1      OPC=nop             
  nop                                    #  168   0xad125  1      OPC=nop             
  nop                                    #  169   0xad126  1      OPC=nop             
.L_ad120:                                #        0xad127  0      OPC=<label>         
  addl $0x1, %eax                        #  170   0xad127  3      OPC=addl_r32_imm8   
  movl %edi, %edi                        #  171   0xad12a  2      OPC=movl_r32_r32    
  movl %eax, 0x8(%r15,%rdi,1)            #  172   0xad12c  5      OPC=movl_m32_r32    
  jmpq .L_ad060                          #  173   0xad131  5      OPC=jmpq_label_1    
  xchgw %ax, %ax                         #  174   0xad136  2      OPC=xchgw_ax_r16    
  nop                                    #  175   0xad138  1      OPC=nop             
  nop                                    #  176   0xad139  1      OPC=nop             
  nop                                    #  177   0xad13a  1      OPC=nop             
  nop                                    #  178   0xad13b  1      OPC=nop             
  nop                                    #  179   0xad13c  1      OPC=nop             
  nop                                    #  180   0xad13d  1      OPC=nop             
  nop                                    #  181   0xad13e  1      OPC=nop             
  nop                                    #  182   0xad13f  1      OPC=nop             
  nop                                    #  183   0xad140  1      OPC=nop             
  nop                                    #  184   0xad141  1      OPC=nop             
  nop                                    #  185   0xad142  1      OPC=nop             
  nop                                    #  186   0xad143  1      OPC=nop             
  nop                                    #  187   0xad144  1      OPC=nop             
  nop                                    #  188   0xad145  1      OPC=nop             
  nop                                    #  189   0xad146  1      OPC=nop             
.L_ad140:                                #        0xad147  0      OPC=<label>         
  movl %edi, %edi                        #  190   0xad147  2      OPC=movl_r32_r32    
  movl 0x8(%r15,%rdi,1), %edx            #  191   0xad149  5      OPC=movl_r32_m32    
  leal -0x1(%rdx), %ecx                  #  192   0xad14e  3      OPC=leal_r32_m16    
  testl %edx, %edx                       #  193   0xad151  2      OPC=testl_r32_r32   
  movl %edi, %edi                        #  194   0xad153  2      OPC=movl_r32_r32    
  movl %ecx, 0x8(%r15,%rdi,1)            #  195   0xad155  5      OPC=movl_m32_r32    
  jg .L_ad0a0                            #  196   0xad15a  6      OPC=jg_label_1      
  leal 0x1f(%rsp), %esi                  #  197   0xad160  4      OPC=leal_r32_m16    
  nop                                    #  198   0xad164  1      OPC=nop             
  nop                                    #  199   0xad165  1      OPC=nop             
  nop                                    #  200   0xad166  1      OPC=nop             
  movq %rax, 0x8(%rsp)                   #  201   0xad167  5      OPC=movq_m64_r64    
  nop                                    #  202   0xad16c  1      OPC=nop             
  nop                                    #  203   0xad16d  1      OPC=nop             
  nop                                    #  204   0xad16e  1      OPC=nop             
  nop                                    #  205   0xad16f  1      OPC=nop             
  nop                                    #  206   0xad170  1      OPC=nop             
  nop                                    #  207   0xad171  1      OPC=nop             
  nop                                    #  208   0xad172  1      OPC=nop             
  nop                                    #  209   0xad173  1      OPC=nop             
  nop                                    #  210   0xad174  1      OPC=nop             
  nop                                    #  211   0xad175  1      OPC=nop             
  nop                                    #  212   0xad176  1      OPC=nop             
  nop                                    #  213   0xad177  1      OPC=nop             
  nop                                    #  214   0xad178  1      OPC=nop             
  nop                                    #  215   0xad179  1      OPC=nop             
  nop                                    #  216   0xad17a  1      OPC=nop             
  nop                                    #  217   0xad17b  1      OPC=nop             
  nop                                    #  218   0xad17c  1      OPC=nop             
  nop                                    #  219   0xad17d  1      OPC=nop             
  nop                                    #  220   0xad17e  1      OPC=nop             
  nop                                    #  221   0xad17f  1      OPC=nop             
  nop                                    #  222   0xad180  1      OPC=nop             
  nop                                    #  223   0xad181  1      OPC=nop             
  callq ._ZNSs4_Rep10_M_destroyERKSaIcE  #  224   0xad182  5      OPC=callq_label     
  movq 0x8(%rsp), %rax                   #  225   0xad187  5      OPC=movq_r64_m64    
  jmpq .L_ad0a0                          #  226   0xad18c  5      OPC=jmpq_label_1    
  nop                                    #  227   0xad191  1      OPC=nop             
  nop                                    #  228   0xad192  1      OPC=nop             
  nop                                    #  229   0xad193  1      OPC=nop             
  nop                                    #  230   0xad194  1      OPC=nop             
  nop                                    #  231   0xad195  1      OPC=nop             
  nop                                    #  232   0xad196  1      OPC=nop             
  nop                                    #  233   0xad197  1      OPC=nop             
  nop                                    #  234   0xad198  1      OPC=nop             
  nop                                    #  235   0xad199  1      OPC=nop             
  nop                                    #  236   0xad19a  1      OPC=nop             
  nop                                    #  237   0xad19b  1      OPC=nop             
  nop                                    #  238   0xad19c  1      OPC=nop             
  nop                                    #  239   0xad19d  1      OPC=nop             
  nop                                    #  240   0xad19e  1      OPC=nop             
  nop                                    #  241   0xad19f  1      OPC=nop             
  nop                                    #  242   0xad1a0  1      OPC=nop             
  nop                                    #  243   0xad1a1  1      OPC=nop             
  nop                                    #  244   0xad1a2  1      OPC=nop             
  nop                                    #  245   0xad1a3  1      OPC=nop             
  nop                                    #  246   0xad1a4  1      OPC=nop             
  nop                                    #  247   0xad1a5  1      OPC=nop             
  nop                                    #  248   0xad1a6  1      OPC=nop             
  movl %eax, %edi                        #  249   0xad1a7  2      OPC=movl_r32_r32    
  nop                                    #  250   0xad1a9  1      OPC=nop             
  nop                                    #  251   0xad1aa  1      OPC=nop             
  nop                                    #  252   0xad1ab  1      OPC=nop             
  nop                                    #  253   0xad1ac  1      OPC=nop             
  nop                                    #  254   0xad1ad  1      OPC=nop             
  nop                                    #  255   0xad1ae  1      OPC=nop             
  nop                                    #  256   0xad1af  1      OPC=nop             
  nop                                    #  257   0xad1b0  1      OPC=nop             
  nop                                    #  258   0xad1b1  1      OPC=nop             
  nop                                    #  259   0xad1b2  1      OPC=nop             
  nop                                    #  260   0xad1b3  1      OPC=nop             
  nop                                    #  261   0xad1b4  1      OPC=nop             
  nop                                    #  262   0xad1b5  1      OPC=nop             
  nop                                    #  263   0xad1b6  1      OPC=nop             
  nop                                    #  264   0xad1b7  1      OPC=nop             
  nop                                    #  265   0xad1b8  1      OPC=nop             
  nop                                    #  266   0xad1b9  1      OPC=nop             
  nop                                    #  267   0xad1ba  1      OPC=nop             
  nop                                    #  268   0xad1bb  1      OPC=nop             
  nop                                    #  269   0xad1bc  1      OPC=nop             
  nop                                    #  270   0xad1bd  1      OPC=nop             
  nop                                    #  271   0xad1be  1      OPC=nop             
  nop                                    #  272   0xad1bf  1      OPC=nop             
  nop                                    #  273   0xad1c0  1      OPC=nop             
  nop                                    #  274   0xad1c1  1      OPC=nop             
  callq ._Unwind_Resume                  #  275   0xad1c2  5      OPC=callq_label     
                                                                                      
.size _ZNSs6assignERKSs, .-_ZNSs6assignERKSs

