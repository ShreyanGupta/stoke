  .text
  .globl _ZSt20__throw_out_of_rangePKc
  .type _ZSt20__throw_out_of_rangePKc, @function

#! file-offset 0x127400
#! rip-offset  0xe7400
#! capacity    416 bytes

# Text                                   #  Line  RIP      Bytes  Opcode              
._ZSt20__throw_out_of_rangePKc:          #        0xe7400  0      OPC=<label>         
  pushq %r12                             #  1     0xe7400  2      OPC=pushq_r64_1     
  movl %edi, %esi                        #  2     0xe7402  2      OPC=movl_r32_r32    
  pushq %rbx                             #  3     0xe7404  1      OPC=pushq_r64_1     
  subl $0x28, %esp                       #  4     0xe7405  3      OPC=subl_r32_imm8   
  addq %r15, %rsp                        #  5     0xe7408  3      OPC=addq_r64_r64    
  leal 0x10(%rsp), %ebx                  #  6     0xe740b  4      OPC=leal_r32_m16    
  leal 0x1f(%rsp), %edx                  #  7     0xe740f  4      OPC=leal_r32_m16    
  movl %ebx, %edi                        #  8     0xe7413  2      OPC=movl_r32_r32    
  nop                                    #  9     0xe7415  1      OPC=nop             
  nop                                    #  10    0xe7416  1      OPC=nop             
  nop                                    #  11    0xe7417  1      OPC=nop             
  nop                                    #  12    0xe7418  1      OPC=nop             
  nop                                    #  13    0xe7419  1      OPC=nop             
  nop                                    #  14    0xe741a  1      OPC=nop             
  callq ._ZNSsC1EPKcRKSaIcE              #  15    0xe741b  5      OPC=callq_label     
  movl $0x8, %edi                        #  16    0xe7420  5      OPC=movl_r32_imm32  
  nop                                    #  17    0xe7425  1      OPC=nop             
  nop                                    #  18    0xe7426  1      OPC=nop             
  nop                                    #  19    0xe7427  1      OPC=nop             
  nop                                    #  20    0xe7428  1      OPC=nop             
  nop                                    #  21    0xe7429  1      OPC=nop             
  nop                                    #  22    0xe742a  1      OPC=nop             
  nop                                    #  23    0xe742b  1      OPC=nop             
  nop                                    #  24    0xe742c  1      OPC=nop             
  nop                                    #  25    0xe742d  1      OPC=nop             
  nop                                    #  26    0xe742e  1      OPC=nop             
  nop                                    #  27    0xe742f  1      OPC=nop             
  nop                                    #  28    0xe7430  1      OPC=nop             
  nop                                    #  29    0xe7431  1      OPC=nop             
  nop                                    #  30    0xe7432  1      OPC=nop             
  nop                                    #  31    0xe7433  1      OPC=nop             
  nop                                    #  32    0xe7434  1      OPC=nop             
  nop                                    #  33    0xe7435  1      OPC=nop             
  nop                                    #  34    0xe7436  1      OPC=nop             
  nop                                    #  35    0xe7437  1      OPC=nop             
  nop                                    #  36    0xe7438  1      OPC=nop             
  nop                                    #  37    0xe7439  1      OPC=nop             
  nop                                    #  38    0xe743a  1      OPC=nop             
  callq .__cxa_allocate_exception        #  39    0xe743b  5      OPC=callq_label     
  movl %eax, %r12d                       #  40    0xe7440  3      OPC=movl_r32_r32    
  movl %ebx, %esi                        #  41    0xe7443  2      OPC=movl_r32_r32    
  movl %r12d, %edi                       #  42    0xe7445  3      OPC=movl_r32_r32    
  nop                                    #  43    0xe7448  1      OPC=nop             
  nop                                    #  44    0xe7449  1      OPC=nop             
  nop                                    #  45    0xe744a  1      OPC=nop             
  nop                                    #  46    0xe744b  1      OPC=nop             
  nop                                    #  47    0xe744c  1      OPC=nop             
  nop                                    #  48    0xe744d  1      OPC=nop             
  nop                                    #  49    0xe744e  1      OPC=nop             
  nop                                    #  50    0xe744f  1      OPC=nop             
  nop                                    #  51    0xe7450  1      OPC=nop             
  nop                                    #  52    0xe7451  1      OPC=nop             
  nop                                    #  53    0xe7452  1      OPC=nop             
  nop                                    #  54    0xe7453  1      OPC=nop             
  nop                                    #  55    0xe7454  1      OPC=nop             
  nop                                    #  56    0xe7455  1      OPC=nop             
  nop                                    #  57    0xe7456  1      OPC=nop             
  nop                                    #  58    0xe7457  1      OPC=nop             
  nop                                    #  59    0xe7458  1      OPC=nop             
  nop                                    #  60    0xe7459  1      OPC=nop             
  nop                                    #  61    0xe745a  1      OPC=nop             
  callq ._ZNSt12out_of_rangeC1ERKSs      #  62    0xe745b  5      OPC=callq_label     
  movl 0x10(%rsp), %edi                  #  63    0xe7460  4      OPC=movl_r32_m32    
  subl $0xc, %edi                        #  64    0xe7464  3      OPC=subl_r32_imm8   
  cmpl $0x10073580, %edi                 #  65    0xe7467  6      OPC=cmpl_r32_imm32  
  jne .L_e74a0                           #  66    0xe746d  2      OPC=jne_label       
  xchgw %ax, %ax                         #  67    0xe746f  2      OPC=xchgw_ax_r16    
  nop                                    #  68    0xe7471  1      OPC=nop             
  nop                                    #  69    0xe7472  1      OPC=nop             
  nop                                    #  70    0xe7473  1      OPC=nop             
  nop                                    #  71    0xe7474  1      OPC=nop             
  nop                                    #  72    0xe7475  1      OPC=nop             
  nop                                    #  73    0xe7476  1      OPC=nop             
  nop                                    #  74    0xe7477  1      OPC=nop             
  nop                                    #  75    0xe7478  1      OPC=nop             
  nop                                    #  76    0xe7479  1      OPC=nop             
  nop                                    #  77    0xe747a  1      OPC=nop             
  nop                                    #  78    0xe747b  1      OPC=nop             
  nop                                    #  79    0xe747c  1      OPC=nop             
  nop                                    #  80    0xe747d  1      OPC=nop             
  nop                                    #  81    0xe747e  1      OPC=nop             
  nop                                    #  82    0xe747f  1      OPC=nop             
.L_e7480:                                #        0xe7480  0      OPC=<label>         
  movl $0xe6960, %edx                    #  83    0xe7480  5      OPC=movl_r32_imm32  
  movl $0x1003d754, %esi                 #  84    0xe7485  5      OPC=movl_r32_imm32  
  movl %r12d, %edi                       #  85    0xe748a  3      OPC=movl_r32_r32    
  nop                                    #  86    0xe748d  1      OPC=nop             
  nop                                    #  87    0xe748e  1      OPC=nop             
  nop                                    #  88    0xe748f  1      OPC=nop             
  nop                                    #  89    0xe7490  1      OPC=nop             
  nop                                    #  90    0xe7491  1      OPC=nop             
  nop                                    #  91    0xe7492  1      OPC=nop             
  nop                                    #  92    0xe7493  1      OPC=nop             
  nop                                    #  93    0xe7494  1      OPC=nop             
  nop                                    #  94    0xe7495  1      OPC=nop             
  nop                                    #  95    0xe7496  1      OPC=nop             
  nop                                    #  96    0xe7497  1      OPC=nop             
  nop                                    #  97    0xe7498  1      OPC=nop             
  nop                                    #  98    0xe7499  1      OPC=nop             
  nop                                    #  99    0xe749a  1      OPC=nop             
  callq .__cxa_throw                     #  100   0xe749b  5      OPC=callq_label     
.L_e74a0:                                #        0xe74a0  0      OPC=<label>         
  movl %edi, %edi                        #  101   0xe74a0  2      OPC=movl_r32_r32    
  movl 0x8(%r15,%rdi,1), %eax            #  102   0xe74a2  5      OPC=movl_r32_m32    
  leal -0x1(%rax), %edx                  #  103   0xe74a7  3      OPC=leal_r32_m16    
  testl %eax, %eax                       #  104   0xe74aa  2      OPC=testl_r32_r32   
  movl %edi, %edi                        #  105   0xe74ac  2      OPC=movl_r32_r32    
  movl %edx, 0x8(%r15,%rdi,1)            #  106   0xe74ae  5      OPC=movl_m32_r32    
  jg .L_e7480                            #  107   0xe74b3  2      OPC=jg_label        
  leal 0x1e(%rsp), %esi                  #  108   0xe74b5  4      OPC=leal_r32_m16    
  xchgw %ax, %ax                         #  109   0xe74b9  2      OPC=xchgw_ax_r16    
  callq ._ZNSs4_Rep10_M_destroyERKSaIcE  #  110   0xe74bb  5      OPC=callq_label     
  jmpq .L_e7480                          #  111   0xe74c0  2      OPC=jmpq_label      
  nop                                    #  112   0xe74c2  1      OPC=nop             
  nop                                    #  113   0xe74c3  1      OPC=nop             
  nop                                    #  114   0xe74c4  1      OPC=nop             
  nop                                    #  115   0xe74c5  1      OPC=nop             
  nop                                    #  116   0xe74c6  1      OPC=nop             
  nop                                    #  117   0xe74c7  1      OPC=nop             
  nop                                    #  118   0xe74c8  1      OPC=nop             
  nop                                    #  119   0xe74c9  1      OPC=nop             
  nop                                    #  120   0xe74ca  1      OPC=nop             
  nop                                    #  121   0xe74cb  1      OPC=nop             
  nop                                    #  122   0xe74cc  1      OPC=nop             
  nop                                    #  123   0xe74cd  1      OPC=nop             
  nop                                    #  124   0xe74ce  1      OPC=nop             
  nop                                    #  125   0xe74cf  1      OPC=nop             
  nop                                    #  126   0xe74d0  1      OPC=nop             
  nop                                    #  127   0xe74d1  1      OPC=nop             
  nop                                    #  128   0xe74d2  1      OPC=nop             
  nop                                    #  129   0xe74d3  1      OPC=nop             
  nop                                    #  130   0xe74d4  1      OPC=nop             
  nop                                    #  131   0xe74d5  1      OPC=nop             
  nop                                    #  132   0xe74d6  1      OPC=nop             
  nop                                    #  133   0xe74d7  1      OPC=nop             
  nop                                    #  134   0xe74d8  1      OPC=nop             
  nop                                    #  135   0xe74d9  1      OPC=nop             
  nop                                    #  136   0xe74da  1      OPC=nop             
  nop                                    #  137   0xe74db  1      OPC=nop             
  nop                                    #  138   0xe74dc  1      OPC=nop             
  nop                                    #  139   0xe74dd  1      OPC=nop             
  nop                                    #  140   0xe74de  1      OPC=nop             
  nop                                    #  141   0xe74df  1      OPC=nop             
  movl %r12d, %edi                       #  142   0xe74e0  3      OPC=movl_r32_r32    
  movl %eax, 0x8(%rsp)                   #  143   0xe74e3  4      OPC=movl_m32_r32    
  nop                                    #  144   0xe74e7  1      OPC=nop             
  nop                                    #  145   0xe74e8  1      OPC=nop             
  nop                                    #  146   0xe74e9  1      OPC=nop             
  nop                                    #  147   0xe74ea  1      OPC=nop             
  nop                                    #  148   0xe74eb  1      OPC=nop             
  nop                                    #  149   0xe74ec  1      OPC=nop             
  nop                                    #  150   0xe74ed  1      OPC=nop             
  nop                                    #  151   0xe74ee  1      OPC=nop             
  nop                                    #  152   0xe74ef  1      OPC=nop             
  nop                                    #  153   0xe74f0  1      OPC=nop             
  nop                                    #  154   0xe74f1  1      OPC=nop             
  nop                                    #  155   0xe74f2  1      OPC=nop             
  nop                                    #  156   0xe74f3  1      OPC=nop             
  nop                                    #  157   0xe74f4  1      OPC=nop             
  nop                                    #  158   0xe74f5  1      OPC=nop             
  nop                                    #  159   0xe74f6  1      OPC=nop             
  nop                                    #  160   0xe74f7  1      OPC=nop             
  nop                                    #  161   0xe74f8  1      OPC=nop             
  nop                                    #  162   0xe74f9  1      OPC=nop             
  nop                                    #  163   0xe74fa  1      OPC=nop             
  callq .__cxa_free_exception            #  164   0xe74fb  5      OPC=callq_label     
  movl 0x8(%rsp), %eax                   #  165   0xe7500  4      OPC=movl_r32_m32    
  movl %ebx, %edi                        #  166   0xe7504  2      OPC=movl_r32_r32    
  movl %eax, %r12d                       #  167   0xe7506  3      OPC=movl_r32_r32    
  nop                                    #  168   0xe7509  1      OPC=nop             
  nop                                    #  169   0xe750a  1      OPC=nop             
  nop                                    #  170   0xe750b  1      OPC=nop             
  nop                                    #  171   0xe750c  1      OPC=nop             
  nop                                    #  172   0xe750d  1      OPC=nop             
  nop                                    #  173   0xe750e  1      OPC=nop             
  nop                                    #  174   0xe750f  1      OPC=nop             
  nop                                    #  175   0xe7510  1      OPC=nop             
  nop                                    #  176   0xe7511  1      OPC=nop             
  nop                                    #  177   0xe7512  1      OPC=nop             
  nop                                    #  178   0xe7513  1      OPC=nop             
  nop                                    #  179   0xe7514  1      OPC=nop             
  nop                                    #  180   0xe7515  1      OPC=nop             
  nop                                    #  181   0xe7516  1      OPC=nop             
  nop                                    #  182   0xe7517  1      OPC=nop             
  nop                                    #  183   0xe7518  1      OPC=nop             
  nop                                    #  184   0xe7519  1      OPC=nop             
  nop                                    #  185   0xe751a  1      OPC=nop             
  callq ._ZNSsD1Ev                       #  186   0xe751b  5      OPC=callq_label     
  movl %r12d, %eax                       #  187   0xe7520  3      OPC=movl_r32_r32    
  nop                                    #  188   0xe7523  1      OPC=nop             
  nop                                    #  189   0xe7524  1      OPC=nop             
  nop                                    #  190   0xe7525  1      OPC=nop             
  nop                                    #  191   0xe7526  1      OPC=nop             
  nop                                    #  192   0xe7527  1      OPC=nop             
  nop                                    #  193   0xe7528  1      OPC=nop             
  nop                                    #  194   0xe7529  1      OPC=nop             
  nop                                    #  195   0xe752a  1      OPC=nop             
  nop                                    #  196   0xe752b  1      OPC=nop             
  nop                                    #  197   0xe752c  1      OPC=nop             
  nop                                    #  198   0xe752d  1      OPC=nop             
  nop                                    #  199   0xe752e  1      OPC=nop             
  nop                                    #  200   0xe752f  1      OPC=nop             
  nop                                    #  201   0xe7530  1      OPC=nop             
  nop                                    #  202   0xe7531  1      OPC=nop             
  nop                                    #  203   0xe7532  1      OPC=nop             
  nop                                    #  204   0xe7533  1      OPC=nop             
  nop                                    #  205   0xe7534  1      OPC=nop             
  nop                                    #  206   0xe7535  1      OPC=nop             
  nop                                    #  207   0xe7536  1      OPC=nop             
  nop                                    #  208   0xe7537  1      OPC=nop             
  nop                                    #  209   0xe7538  1      OPC=nop             
  nop                                    #  210   0xe7539  1      OPC=nop             
  nop                                    #  211   0xe753a  1      OPC=nop             
  nop                                    #  212   0xe753b  1      OPC=nop             
  nop                                    #  213   0xe753c  1      OPC=nop             
  nop                                    #  214   0xe753d  1      OPC=nop             
  nop                                    #  215   0xe753e  1      OPC=nop             
  nop                                    #  216   0xe753f  1      OPC=nop             
.L_e7540:                                #        0xe7540  0      OPC=<label>         
  movl %eax, %edi                        #  217   0xe7540  2      OPC=movl_r32_r32    
  nop                                    #  218   0xe7542  1      OPC=nop             
  nop                                    #  219   0xe7543  1      OPC=nop             
  nop                                    #  220   0xe7544  1      OPC=nop             
  nop                                    #  221   0xe7545  1      OPC=nop             
  nop                                    #  222   0xe7546  1      OPC=nop             
  nop                                    #  223   0xe7547  1      OPC=nop             
  nop                                    #  224   0xe7548  1      OPC=nop             
  nop                                    #  225   0xe7549  1      OPC=nop             
  nop                                    #  226   0xe754a  1      OPC=nop             
  nop                                    #  227   0xe754b  1      OPC=nop             
  nop                                    #  228   0xe754c  1      OPC=nop             
  nop                                    #  229   0xe754d  1      OPC=nop             
  nop                                    #  230   0xe754e  1      OPC=nop             
  nop                                    #  231   0xe754f  1      OPC=nop             
  nop                                    #  232   0xe7550  1      OPC=nop             
  nop                                    #  233   0xe7551  1      OPC=nop             
  nop                                    #  234   0xe7552  1      OPC=nop             
  nop                                    #  235   0xe7553  1      OPC=nop             
  nop                                    #  236   0xe7554  1      OPC=nop             
  nop                                    #  237   0xe7555  1      OPC=nop             
  nop                                    #  238   0xe7556  1      OPC=nop             
  nop                                    #  239   0xe7557  1      OPC=nop             
  nop                                    #  240   0xe7558  1      OPC=nop             
  nop                                    #  241   0xe7559  1      OPC=nop             
  nop                                    #  242   0xe755a  1      OPC=nop             
  callq ._Unwind_Resume                  #  243   0xe755b  5      OPC=callq_label     
  jmpq .L_e7540                          #  244   0xe7560  2      OPC=jmpq_label      
  nop                                    #  245   0xe7562  1      OPC=nop             
  nop                                    #  246   0xe7563  1      OPC=nop             
  nop                                    #  247   0xe7564  1      OPC=nop             
  nop                                    #  248   0xe7565  1      OPC=nop             
  nop                                    #  249   0xe7566  1      OPC=nop             
  nop                                    #  250   0xe7567  1      OPC=nop             
  nop                                    #  251   0xe7568  1      OPC=nop             
  nop                                    #  252   0xe7569  1      OPC=nop             
  nop                                    #  253   0xe756a  1      OPC=nop             
  nop                                    #  254   0xe756b  1      OPC=nop             
  nop                                    #  255   0xe756c  1      OPC=nop             
  nop                                    #  256   0xe756d  1      OPC=nop             
  nop                                    #  257   0xe756e  1      OPC=nop             
  nop                                    #  258   0xe756f  1      OPC=nop             
  nop                                    #  259   0xe7570  1      OPC=nop             
  nop                                    #  260   0xe7571  1      OPC=nop             
  nop                                    #  261   0xe7572  1      OPC=nop             
  nop                                    #  262   0xe7573  1      OPC=nop             
  nop                                    #  263   0xe7574  1      OPC=nop             
  nop                                    #  264   0xe7575  1      OPC=nop             
  nop                                    #  265   0xe7576  1      OPC=nop             
  nop                                    #  266   0xe7577  1      OPC=nop             
  nop                                    #  267   0xe7578  1      OPC=nop             
  nop                                    #  268   0xe7579  1      OPC=nop             
  nop                                    #  269   0xe757a  1      OPC=nop             
  nop                                    #  270   0xe757b  1      OPC=nop             
  nop                                    #  271   0xe757c  1      OPC=nop             
  nop                                    #  272   0xe757d  1      OPC=nop             
  nop                                    #  273   0xe757e  1      OPC=nop             
  nop                                    #  274   0xe757f  1      OPC=nop             
  cmpq $0xff, %rdx                       #  275   0xe7580  4      OPC=cmpq_r64_imm8   
  jne .L_e7540                           #  276   0xe7584  2      OPC=jne_label       
  nop                                    #  277   0xe7586  1      OPC=nop             
  nop                                    #  278   0xe7587  1      OPC=nop             
  nop                                    #  279   0xe7588  1      OPC=nop             
  nop                                    #  280   0xe7589  1      OPC=nop             
  nop                                    #  281   0xe758a  1      OPC=nop             
  nop                                    #  282   0xe758b  1      OPC=nop             
  nop                                    #  283   0xe758c  1      OPC=nop             
  nop                                    #  284   0xe758d  1      OPC=nop             
  nop                                    #  285   0xe758e  1      OPC=nop             
  nop                                    #  286   0xe758f  1      OPC=nop             
  nop                                    #  287   0xe7590  1      OPC=nop             
  nop                                    #  288   0xe7591  1      OPC=nop             
  nop                                    #  289   0xe7592  1      OPC=nop             
  nop                                    #  290   0xe7593  1      OPC=nop             
  nop                                    #  291   0xe7594  1      OPC=nop             
  nop                                    #  292   0xe7595  1      OPC=nop             
  nop                                    #  293   0xe7596  1      OPC=nop             
  nop                                    #  294   0xe7597  1      OPC=nop             
  nop                                    #  295   0xe7598  1      OPC=nop             
  nop                                    #  296   0xe7599  1      OPC=nop             
  nop                                    #  297   0xe759a  1      OPC=nop             
  callq ._ZSt9terminatev                 #  298   0xe759b  5      OPC=callq_label     
                                                                                      
.size _ZSt20__throw_out_of_rangePKc, .-_ZSt20__throw_out_of_rangePKc

