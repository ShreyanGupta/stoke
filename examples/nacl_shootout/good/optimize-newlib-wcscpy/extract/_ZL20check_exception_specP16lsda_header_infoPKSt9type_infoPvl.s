  .text
  .globl _ZL20check_exception_specP16lsda_header_infoPKSt9type_infoPvl
  .type _ZL20check_exception_specP16lsda_header_infoPKSt9type_infoPvl, @function

#! file-offset 0x11f600
#! rip-offset  0xdf600
#! capacity    288 bytes

# Text                                                           #  Line  RIP      Bytes  Opcode              
._ZL20check_exception_specP16lsda_header_infoPKSt9type_infoPvl:  #        0xdf600  0      OPC=<label>         
  pushq %r14                                                     #  1     0xdf600  2      OPC=pushq_r64_1     
  notl %ecx                                                      #  2     0xdf602  2      OPC=notl_r32        
  pushq %r13                                                     #  3     0xdf604  2      OPC=pushq_r64_1     
  movl %esi, %r13d                                               #  4     0xdf606  3      OPC=movl_r32_r32    
  xorl %esi, %esi                                                #  5     0xdf609  2      OPC=xorl_r32_r32    
  pushq %r12                                                     #  6     0xdf60b  2      OPC=pushq_r64_1     
  movl %edi, %r12d                                               #  7     0xdf60d  3      OPC=movl_r32_r32    
  pushq %rbx                                                     #  8     0xdf610  1      OPC=pushq_r64_1     
  subl $0x18, %esp                                               #  9     0xdf611  3      OPC=subl_r32_imm8   
  addq %r15, %rsp                                                #  10    0xdf614  3      OPC=addq_r64_r64    
  movl %r12d, %r12d                                              #  11    0xdf617  3      OPC=movl_r32_r32    
  movl 0xc(%r15,%r12,1), %ebx                                    #  12    0xdf61a  5      OPC=movl_r32_m32    
  nop                                                            #  13    0xdf61f  1      OPC=nop             
  leal 0xc(%rsp), %r14d                                          #  14    0xdf620  5      OPC=leal_r32_m16    
  movl %edx, 0xc(%rsp)                                           #  15    0xdf625  4      OPC=movl_m32_r32    
  addl %ecx, %ebx                                                #  16    0xdf629  2      OPC=addl_r32_r32    
  xorl %ecx, %ecx                                                #  17    0xdf62b  2      OPC=xorl_r32_r32    
  nop                                                            #  18    0xdf62d  1      OPC=nop             
  nop                                                            #  19    0xdf62e  1      OPC=nop             
  nop                                                            #  20    0xdf62f  1      OPC=nop             
  nop                                                            #  21    0xdf630  1      OPC=nop             
  nop                                                            #  22    0xdf631  1      OPC=nop             
  nop                                                            #  23    0xdf632  1      OPC=nop             
  nop                                                            #  24    0xdf633  1      OPC=nop             
  nop                                                            #  25    0xdf634  1      OPC=nop             
  nop                                                            #  26    0xdf635  1      OPC=nop             
  nop                                                            #  27    0xdf636  1      OPC=nop             
  nop                                                            #  28    0xdf637  1      OPC=nop             
  nop                                                            #  29    0xdf638  1      OPC=nop             
  nop                                                            #  30    0xdf639  1      OPC=nop             
  nop                                                            #  31    0xdf63a  1      OPC=nop             
  nop                                                            #  32    0xdf63b  1      OPC=nop             
  nop                                                            #  33    0xdf63c  1      OPC=nop             
  nop                                                            #  34    0xdf63d  1      OPC=nop             
  nop                                                            #  35    0xdf63e  1      OPC=nop             
  nop                                                            #  36    0xdf63f  1      OPC=nop             
.L_df640:                                                        #        0xdf640  0      OPC=<label>         
  movl %ebx, %ebx                                                #  37    0xdf640  2      OPC=movl_r32_r32    
  movzbl (%r15,%rbx,1), %edx                                     #  38    0xdf642  5      OPC=movzbl_r32_m8   
  addl $0x1, %ebx                                                #  39    0xdf647  3      OPC=addl_r32_imm8   
  movl %edx, %eax                                                #  40    0xdf64a  2      OPC=movl_r32_r32    
  andl $0x7f, %eax                                               #  41    0xdf64c  3      OPC=andl_r32_imm8   
  shll %cl, %eax                                                 #  42    0xdf64f  2      OPC=shll_r32_cl     
  orl %eax, %esi                                                 #  43    0xdf651  2      OPC=orl_r32_r32     
  testb %dl, %dl                                                 #  44    0xdf653  2      OPC=testb_r8_r8     
  js .L_df6c0                                                    #  45    0xdf655  2      OPC=js_label        
  testl %esi, %esi                                               #  46    0xdf657  2      OPC=testl_r32_r32   
  je .L_df6e0                                                    #  47    0xdf659  6      OPC=je_label_1      
  nop                                                            #  48    0xdf65f  1      OPC=nop             
  movl %r12d, %edi                                               #  49    0xdf660  3      OPC=movl_r32_r32    
  nop                                                            #  50    0xdf663  1      OPC=nop             
  nop                                                            #  51    0xdf664  1      OPC=nop             
  nop                                                            #  52    0xdf665  1      OPC=nop             
  nop                                                            #  53    0xdf666  1      OPC=nop             
  nop                                                            #  54    0xdf667  1      OPC=nop             
  nop                                                            #  55    0xdf668  1      OPC=nop             
  nop                                                            #  56    0xdf669  1      OPC=nop             
  nop                                                            #  57    0xdf66a  1      OPC=nop             
  nop                                                            #  58    0xdf66b  1      OPC=nop             
  nop                                                            #  59    0xdf66c  1      OPC=nop             
  nop                                                            #  60    0xdf66d  1      OPC=nop             
  nop                                                            #  61    0xdf66e  1      OPC=nop             
  nop                                                            #  62    0xdf66f  1      OPC=nop             
  nop                                                            #  63    0xdf670  1      OPC=nop             
  nop                                                            #  64    0xdf671  1      OPC=nop             
  nop                                                            #  65    0xdf672  1      OPC=nop             
  nop                                                            #  66    0xdf673  1      OPC=nop             
  nop                                                            #  67    0xdf674  1      OPC=nop             
  nop                                                            #  68    0xdf675  1      OPC=nop             
  nop                                                            #  69    0xdf676  1      OPC=nop             
  nop                                                            #  70    0xdf677  1      OPC=nop             
  nop                                                            #  71    0xdf678  1      OPC=nop             
  nop                                                            #  72    0xdf679  1      OPC=nop             
  nop                                                            #  73    0xdf67a  1      OPC=nop             
  callq ._ZL15get_ttype_entryP16lsda_header_infom                #  74    0xdf67b  5      OPC=callq_label     
  movl %r14d, %edx                                               #  75    0xdf680  3      OPC=movl_r32_r32    
  movl %eax, %edi                                                #  76    0xdf683  2      OPC=movl_r32_r32    
  movl %r13d, %esi                                               #  77    0xdf685  3      OPC=movl_r32_r32    
  nop                                                            #  78    0xdf688  1      OPC=nop             
  nop                                                            #  79    0xdf689  1      OPC=nop             
  nop                                                            #  80    0xdf68a  1      OPC=nop             
  nop                                                            #  81    0xdf68b  1      OPC=nop             
  nop                                                            #  82    0xdf68c  1      OPC=nop             
  nop                                                            #  83    0xdf68d  1      OPC=nop             
  nop                                                            #  84    0xdf68e  1      OPC=nop             
  nop                                                            #  85    0xdf68f  1      OPC=nop             
  nop                                                            #  86    0xdf690  1      OPC=nop             
  nop                                                            #  87    0xdf691  1      OPC=nop             
  nop                                                            #  88    0xdf692  1      OPC=nop             
  nop                                                            #  89    0xdf693  1      OPC=nop             
  nop                                                            #  90    0xdf694  1      OPC=nop             
  nop                                                            #  91    0xdf695  1      OPC=nop             
  nop                                                            #  92    0xdf696  1      OPC=nop             
  nop                                                            #  93    0xdf697  1      OPC=nop             
  nop                                                            #  94    0xdf698  1      OPC=nop             
  nop                                                            #  95    0xdf699  1      OPC=nop             
  nop                                                            #  96    0xdf69a  1      OPC=nop             
  callq ._ZL16get_adjusted_ptrPKSt9type_infoS1_PPv               #  97    0xdf69b  5      OPC=callq_label     
  testb %al, %al                                                 #  98    0xdf6a0  2      OPC=testb_r8_r8     
  jne .L_df700                                                   #  99    0xdf6a2  2      OPC=jne_label       
  xorl %esi, %esi                                                #  100   0xdf6a4  2      OPC=xorl_r32_r32    
  xorl %ecx, %ecx                                                #  101   0xdf6a6  2      OPC=xorl_r32_r32    
  jmpq .L_df640                                                  #  102   0xdf6a8  2      OPC=jmpq_label      
  nop                                                            #  103   0xdf6aa  1      OPC=nop             
  nop                                                            #  104   0xdf6ab  1      OPC=nop             
  nop                                                            #  105   0xdf6ac  1      OPC=nop             
  nop                                                            #  106   0xdf6ad  1      OPC=nop             
  nop                                                            #  107   0xdf6ae  1      OPC=nop             
  nop                                                            #  108   0xdf6af  1      OPC=nop             
  nop                                                            #  109   0xdf6b0  1      OPC=nop             
  nop                                                            #  110   0xdf6b1  1      OPC=nop             
  nop                                                            #  111   0xdf6b2  1      OPC=nop             
  nop                                                            #  112   0xdf6b3  1      OPC=nop             
  nop                                                            #  113   0xdf6b4  1      OPC=nop             
  nop                                                            #  114   0xdf6b5  1      OPC=nop             
  nop                                                            #  115   0xdf6b6  1      OPC=nop             
  nop                                                            #  116   0xdf6b7  1      OPC=nop             
  nop                                                            #  117   0xdf6b8  1      OPC=nop             
  nop                                                            #  118   0xdf6b9  1      OPC=nop             
  nop                                                            #  119   0xdf6ba  1      OPC=nop             
  nop                                                            #  120   0xdf6bb  1      OPC=nop             
  nop                                                            #  121   0xdf6bc  1      OPC=nop             
  nop                                                            #  122   0xdf6bd  1      OPC=nop             
  nop                                                            #  123   0xdf6be  1      OPC=nop             
  nop                                                            #  124   0xdf6bf  1      OPC=nop             
.L_df6c0:                                                        #        0xdf6c0  0      OPC=<label>         
  addl $0x7, %ecx                                                #  125   0xdf6c0  3      OPC=addl_r32_imm8   
  jmpq .L_df640                                                  #  126   0xdf6c3  5      OPC=jmpq_label_1    
  nop                                                            #  127   0xdf6c8  1      OPC=nop             
  nop                                                            #  128   0xdf6c9  1      OPC=nop             
  nop                                                            #  129   0xdf6ca  1      OPC=nop             
  nop                                                            #  130   0xdf6cb  1      OPC=nop             
  nop                                                            #  131   0xdf6cc  1      OPC=nop             
  nop                                                            #  132   0xdf6cd  1      OPC=nop             
  nop                                                            #  133   0xdf6ce  1      OPC=nop             
  nop                                                            #  134   0xdf6cf  1      OPC=nop             
  nop                                                            #  135   0xdf6d0  1      OPC=nop             
  nop                                                            #  136   0xdf6d1  1      OPC=nop             
  nop                                                            #  137   0xdf6d2  1      OPC=nop             
  nop                                                            #  138   0xdf6d3  1      OPC=nop             
  nop                                                            #  139   0xdf6d4  1      OPC=nop             
  nop                                                            #  140   0xdf6d5  1      OPC=nop             
  nop                                                            #  141   0xdf6d6  1      OPC=nop             
  nop                                                            #  142   0xdf6d7  1      OPC=nop             
  nop                                                            #  143   0xdf6d8  1      OPC=nop             
  nop                                                            #  144   0xdf6d9  1      OPC=nop             
  nop                                                            #  145   0xdf6da  1      OPC=nop             
  nop                                                            #  146   0xdf6db  1      OPC=nop             
  nop                                                            #  147   0xdf6dc  1      OPC=nop             
  nop                                                            #  148   0xdf6dd  1      OPC=nop             
  nop                                                            #  149   0xdf6de  1      OPC=nop             
  nop                                                            #  150   0xdf6df  1      OPC=nop             
.L_df6e0:                                                        #        0xdf6e0  0      OPC=<label>         
  addl $0x18, %esp                                               #  151   0xdf6e0  3      OPC=addl_r32_imm8   
  addq %r15, %rsp                                                #  152   0xdf6e3  3      OPC=addq_r64_r64    
  xorl %eax, %eax                                                #  153   0xdf6e6  2      OPC=xorl_r32_r32    
  popq %rbx                                                      #  154   0xdf6e8  1      OPC=popq_r64_1      
  popq %r12                                                      #  155   0xdf6e9  2      OPC=popq_r64_1      
  popq %r13                                                      #  156   0xdf6eb  2      OPC=popq_r64_1      
  popq %r14                                                      #  157   0xdf6ed  2      OPC=popq_r64_1      
  popq %r11                                                      #  158   0xdf6ef  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d                                        #  159   0xdf6f1  7      OPC=andl_r32_imm32  
  nop                                                            #  160   0xdf6f8  1      OPC=nop             
  nop                                                            #  161   0xdf6f9  1      OPC=nop             
  nop                                                            #  162   0xdf6fa  1      OPC=nop             
  nop                                                            #  163   0xdf6fb  1      OPC=nop             
  addq %r15, %r11                                                #  164   0xdf6fc  3      OPC=addq_r64_r64    
  jmpq %r11                                                      #  165   0xdf6ff  3      OPC=jmpq_r64        
  nop                                                            #  166   0xdf702  1      OPC=nop             
  nop                                                            #  167   0xdf703  1      OPC=nop             
  nop                                                            #  168   0xdf704  1      OPC=nop             
  nop                                                            #  169   0xdf705  1      OPC=nop             
  nop                                                            #  170   0xdf706  1      OPC=nop             
.L_df700:                                                        #        0xdf707  0      OPC=<label>         
  addl $0x18, %esp                                               #  171   0xdf707  3      OPC=addl_r32_imm8   
  addq %r15, %rsp                                                #  172   0xdf70a  3      OPC=addq_r64_r64    
  movl $0x1, %eax                                                #  173   0xdf70d  5      OPC=movl_r32_imm32  
  popq %rbx                                                      #  174   0xdf712  1      OPC=popq_r64_1      
  popq %r12                                                      #  175   0xdf713  2      OPC=popq_r64_1      
  popq %r13                                                      #  176   0xdf715  2      OPC=popq_r64_1      
  popq %r14                                                      #  177   0xdf717  2      OPC=popq_r64_1      
  popq %r11                                                      #  178   0xdf719  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d                                        #  179   0xdf71b  7      OPC=andl_r32_imm32  
  nop                                                            #  180   0xdf722  1      OPC=nop             
  nop                                                            #  181   0xdf723  1      OPC=nop             
  nop                                                            #  182   0xdf724  1      OPC=nop             
  nop                                                            #  183   0xdf725  1      OPC=nop             
  addq %r15, %r11                                                #  184   0xdf726  3      OPC=addq_r64_r64    
  jmpq %r11                                                      #  185   0xdf729  3      OPC=jmpq_r64        
  nop                                                            #  186   0xdf72c  1      OPC=nop             
  nop                                                            #  187   0xdf72d  1      OPC=nop             
                                                                                                              
.size _ZL20check_exception_specP16lsda_header_infoPKSt9type_infoPvl, .-_ZL20check_exception_specP16lsda_header_infoPKSt9type_infoPvl

