  .text
  .globl _ZNSt11__timepunctIcED1Ev
  .type _ZNSt11__timepunctIcED1Ev, @function

#! file-offset 0xbcc40
#! rip-offset  0x7cc40
#! capacity    288 bytes

# Text                                                #  Line  RIP      Bytes  Opcode              
._ZNSt11__timepunctIcED1Ev:                           #        0x7cc40  0      OPC=<label>         
  pushq %r12                                          #  1     0x7cc40  2      OPC=pushq_r64_1     
  pushq %rbx                                          #  2     0x7cc42  1      OPC=pushq_r64_1     
  movl %edi, %ebx                                     #  3     0x7cc43  2      OPC=movl_r32_r32    
  subl $0x8, %esp                                     #  4     0x7cc45  3      OPC=subl_r32_imm8   
  addq %r15, %rsp                                     #  5     0x7cc48  3      OPC=addq_r64_r64    
  movl %ebx, %ebx                                     #  6     0x7cc4b  2      OPC=movl_r32_r32    
  movl $0x1003aec8, (%r15,%rbx,1)                     #  7     0x7cc4d  8      OPC=movl_m32_imm32  
  movl %ebx, %ebx                                     #  8     0x7cc55  2      OPC=movl_r32_r32    
  movl 0x10(%r15,%rbx,1), %r12d                       #  9     0x7cc57  5      OPC=movl_r32_m32    
  nop                                                 #  10    0x7cc5c  1      OPC=nop             
  nop                                                 #  11    0x7cc5d  1      OPC=nop             
  nop                                                 #  12    0x7cc5e  1      OPC=nop             
  nop                                                 #  13    0x7cc5f  1      OPC=nop             
  nop                                                 #  14    0x7cc60  1      OPC=nop             
  nop                                                 #  15    0x7cc61  1      OPC=nop             
  nop                                                 #  16    0x7cc62  1      OPC=nop             
  nop                                                 #  17    0x7cc63  1      OPC=nop             
  nop                                                 #  18    0x7cc64  1      OPC=nop             
  nop                                                 #  19    0x7cc65  1      OPC=nop             
  nop                                                 #  20    0x7cc66  1      OPC=nop             
  nop                                                 #  21    0x7cc67  1      OPC=nop             
  nop                                                 #  22    0x7cc68  1      OPC=nop             
  nop                                                 #  23    0x7cc69  1      OPC=nop             
  nop                                                 #  24    0x7cc6a  1      OPC=nop             
  nop                                                 #  25    0x7cc6b  1      OPC=nop             
  nop                                                 #  26    0x7cc6c  1      OPC=nop             
  nop                                                 #  27    0x7cc6d  1      OPC=nop             
  nop                                                 #  28    0x7cc6e  1      OPC=nop             
  nop                                                 #  29    0x7cc6f  1      OPC=nop             
  nop                                                 #  30    0x7cc70  1      OPC=nop             
  nop                                                 #  31    0x7cc71  1      OPC=nop             
  nop                                                 #  32    0x7cc72  1      OPC=nop             
  nop                                                 #  33    0x7cc73  1      OPC=nop             
  nop                                                 #  34    0x7cc74  1      OPC=nop             
  nop                                                 #  35    0x7cc75  1      OPC=nop             
  nop                                                 #  36    0x7cc76  1      OPC=nop             
  nop                                                 #  37    0x7cc77  1      OPC=nop             
  nop                                                 #  38    0x7cc78  1      OPC=nop             
  nop                                                 #  39    0x7cc79  1      OPC=nop             
  nop                                                 #  40    0x7cc7a  1      OPC=nop             
  callq ._ZNSt6locale5facet13_S_get_c_nameEv          #  41    0x7cc7b  5      OPC=callq_label     
  cmpl %r12d, %eax                                    #  42    0x7cc80  3      OPC=cmpl_r32_r32    
  je .L_7cca0                                         #  43    0x7cc83  2      OPC=je_label        
  movl %ebx, %ebx                                     #  44    0x7cc85  2      OPC=movl_r32_r32    
  movl 0x10(%r15,%rbx,1), %edi                        #  45    0x7cc87  5      OPC=movl_r32_m32    
  testq %rdi, %rdi                                    #  46    0x7cc8c  3      OPC=testq_r64_r64   
  je .L_7cca0                                         #  47    0x7cc8f  2      OPC=je_label        
  nop                                                 #  48    0x7cc91  1      OPC=nop             
  nop                                                 #  49    0x7cc92  1      OPC=nop             
  nop                                                 #  50    0x7cc93  1      OPC=nop             
  nop                                                 #  51    0x7cc94  1      OPC=nop             
  nop                                                 #  52    0x7cc95  1      OPC=nop             
  nop                                                 #  53    0x7cc96  1      OPC=nop             
  nop                                                 #  54    0x7cc97  1      OPC=nop             
  nop                                                 #  55    0x7cc98  1      OPC=nop             
  nop                                                 #  56    0x7cc99  1      OPC=nop             
  nop                                                 #  57    0x7cc9a  1      OPC=nop             
  callq ._ZdaPv                                       #  58    0x7cc9b  5      OPC=callq_label     
.L_7cca0:                                             #        0x7cca0  0      OPC=<label>         
  movl %ebx, %ebx                                     #  59    0x7cca0  2      OPC=movl_r32_r32    
  movl 0x8(%r15,%rbx,1), %edi                         #  60    0x7cca2  5      OPC=movl_r32_m32    
  testq %rdi, %rdi                                    #  61    0x7cca7  3      OPC=testq_r64_r64   
  je .L_7cce0                                         #  62    0x7ccaa  2      OPC=je_label        
  movl %edi, %edi                                     #  63    0x7ccac  2      OPC=movl_r32_r32    
  movl (%r15,%rdi,1), %eax                            #  64    0x7ccae  4      OPC=movl_r32_m32    
  movl %eax, %eax                                     #  65    0x7ccb2  2      OPC=movl_r32_r32    
  movl 0x4(%r15,%rax,1), %eax                         #  66    0x7ccb4  5      OPC=movl_r32_m32    
  nop                                                 #  67    0x7ccb9  1      OPC=nop             
  nop                                                 #  68    0x7ccba  1      OPC=nop             
  nop                                                 #  69    0x7ccbb  1      OPC=nop             
  nop                                                 #  70    0x7ccbc  1      OPC=nop             
  nop                                                 #  71    0x7ccbd  1      OPC=nop             
  nop                                                 #  72    0x7ccbe  1      OPC=nop             
  nop                                                 #  73    0x7ccbf  1      OPC=nop             
  nop                                                 #  74    0x7ccc0  1      OPC=nop             
  nop                                                 #  75    0x7ccc1  1      OPC=nop             
  nop                                                 #  76    0x7ccc2  1      OPC=nop             
  nop                                                 #  77    0x7ccc3  1      OPC=nop             
  nop                                                 #  78    0x7ccc4  1      OPC=nop             
  nop                                                 #  79    0x7ccc5  1      OPC=nop             
  nop                                                 #  80    0x7ccc6  1      OPC=nop             
  nop                                                 #  81    0x7ccc7  1      OPC=nop             
  nop                                                 #  82    0x7ccc8  1      OPC=nop             
  nop                                                 #  83    0x7ccc9  1      OPC=nop             
  nop                                                 #  84    0x7ccca  1      OPC=nop             
  nop                                                 #  85    0x7cccb  1      OPC=nop             
  nop                                                 #  86    0x7cccc  1      OPC=nop             
  nop                                                 #  87    0x7cccd  1      OPC=nop             
  nop                                                 #  88    0x7ccce  1      OPC=nop             
  nop                                                 #  89    0x7cccf  1      OPC=nop             
  nop                                                 #  90    0x7ccd0  1      OPC=nop             
  nop                                                 #  91    0x7ccd1  1      OPC=nop             
  nop                                                 #  92    0x7ccd2  1      OPC=nop             
  nop                                                 #  93    0x7ccd3  1      OPC=nop             
  nop                                                 #  94    0x7ccd4  1      OPC=nop             
  nop                                                 #  95    0x7ccd5  1      OPC=nop             
  nop                                                 #  96    0x7ccd6  1      OPC=nop             
  nop                                                 #  97    0x7ccd7  1      OPC=nop             
  andl $0xffffffe0, %eax                              #  98    0x7ccd8  6      OPC=andl_r32_imm32  
  nop                                                 #  99    0x7ccde  1      OPC=nop             
  nop                                                 #  100   0x7ccdf  1      OPC=nop             
  nop                                                 #  101   0x7cce0  1      OPC=nop             
  addq %r15, %rax                                     #  102   0x7cce1  3      OPC=addq_r64_r64    
  callq %rax                                          #  103   0x7cce4  2      OPC=callq_r64       
.L_7cce0:                                             #        0x7cce6  0      OPC=<label>         
  leal 0xc(%rbx), %edi                                #  104   0x7cce6  3      OPC=leal_r32_m16    
  nop                                                 #  105   0x7cce9  1      OPC=nop             
  nop                                                 #  106   0x7ccea  1      OPC=nop             
  nop                                                 #  107   0x7cceb  1      OPC=nop             
  nop                                                 #  108   0x7ccec  1      OPC=nop             
  nop                                                 #  109   0x7cced  1      OPC=nop             
  nop                                                 #  110   0x7ccee  1      OPC=nop             
  nop                                                 #  111   0x7ccef  1      OPC=nop             
  nop                                                 #  112   0x7ccf0  1      OPC=nop             
  nop                                                 #  113   0x7ccf1  1      OPC=nop             
  nop                                                 #  114   0x7ccf2  1      OPC=nop             
  nop                                                 #  115   0x7ccf3  1      OPC=nop             
  nop                                                 #  116   0x7ccf4  1      OPC=nop             
  nop                                                 #  117   0x7ccf5  1      OPC=nop             
  nop                                                 #  118   0x7ccf6  1      OPC=nop             
  nop                                                 #  119   0x7ccf7  1      OPC=nop             
  nop                                                 #  120   0x7ccf8  1      OPC=nop             
  nop                                                 #  121   0x7ccf9  1      OPC=nop             
  nop                                                 #  122   0x7ccfa  1      OPC=nop             
  nop                                                 #  123   0x7ccfb  1      OPC=nop             
  nop                                                 #  124   0x7ccfc  1      OPC=nop             
  nop                                                 #  125   0x7ccfd  1      OPC=nop             
  nop                                                 #  126   0x7ccfe  1      OPC=nop             
  nop                                                 #  127   0x7ccff  1      OPC=nop             
  nop                                                 #  128   0x7cd00  1      OPC=nop             
  callq ._ZNSt6locale5facet19_S_destroy_c_localeERPi  #  129   0x7cd01  5      OPC=callq_label     
  addl $0x8, %esp                                     #  130   0x7cd06  3      OPC=addl_r32_imm8   
  addq %r15, %rsp                                     #  131   0x7cd09  3      OPC=addq_r64_r64    
  movl %ebx, %edi                                     #  132   0x7cd0c  2      OPC=movl_r32_r32    
  popq %rbx                                           #  133   0x7cd0e  1      OPC=popq_r64_1      
  popq %r12                                           #  134   0x7cd0f  2      OPC=popq_r64_1      
  jmpq ._ZNSt6locale5facetD2Ev                        #  135   0x7cd11  5      OPC=jmpq_label_1    
  nop                                                 #  136   0x7cd16  1      OPC=nop             
  nop                                                 #  137   0x7cd17  1      OPC=nop             
  nop                                                 #  138   0x7cd18  1      OPC=nop             
  nop                                                 #  139   0x7cd19  1      OPC=nop             
  nop                                                 #  140   0x7cd1a  1      OPC=nop             
  nop                                                 #  141   0x7cd1b  1      OPC=nop             
  nop                                                 #  142   0x7cd1c  1      OPC=nop             
  nop                                                 #  143   0x7cd1d  1      OPC=nop             
  nop                                                 #  144   0x7cd1e  1      OPC=nop             
  nop                                                 #  145   0x7cd1f  1      OPC=nop             
  nop                                                 #  146   0x7cd20  1      OPC=nop             
  nop                                                 #  147   0x7cd21  1      OPC=nop             
  nop                                                 #  148   0x7cd22  1      OPC=nop             
  nop                                                 #  149   0x7cd23  1      OPC=nop             
  nop                                                 #  150   0x7cd24  1      OPC=nop             
  nop                                                 #  151   0x7cd25  1      OPC=nop             
  movl %eax, %r12d                                    #  152   0x7cd26  3      OPC=movl_r32_r32    
  movl %ebx, %edi                                     #  153   0x7cd29  2      OPC=movl_r32_r32    
  nop                                                 #  154   0x7cd2b  1      OPC=nop             
  nop                                                 #  155   0x7cd2c  1      OPC=nop             
  nop                                                 #  156   0x7cd2d  1      OPC=nop             
  nop                                                 #  157   0x7cd2e  1      OPC=nop             
  nop                                                 #  158   0x7cd2f  1      OPC=nop             
  nop                                                 #  159   0x7cd30  1      OPC=nop             
  nop                                                 #  160   0x7cd31  1      OPC=nop             
  nop                                                 #  161   0x7cd32  1      OPC=nop             
  nop                                                 #  162   0x7cd33  1      OPC=nop             
  nop                                                 #  163   0x7cd34  1      OPC=nop             
  nop                                                 #  164   0x7cd35  1      OPC=nop             
  nop                                                 #  165   0x7cd36  1      OPC=nop             
  nop                                                 #  166   0x7cd37  1      OPC=nop             
  nop                                                 #  167   0x7cd38  1      OPC=nop             
  nop                                                 #  168   0x7cd39  1      OPC=nop             
  nop                                                 #  169   0x7cd3a  1      OPC=nop             
  nop                                                 #  170   0x7cd3b  1      OPC=nop             
  nop                                                 #  171   0x7cd3c  1      OPC=nop             
  nop                                                 #  172   0x7cd3d  1      OPC=nop             
  nop                                                 #  173   0x7cd3e  1      OPC=nop             
  nop                                                 #  174   0x7cd3f  1      OPC=nop             
  nop                                                 #  175   0x7cd40  1      OPC=nop             
  callq ._ZNSt6locale5facetD2Ev                       #  176   0x7cd41  5      OPC=callq_label     
  movl %r12d, %edi                                    #  177   0x7cd46  3      OPC=movl_r32_r32    
  nop                                                 #  178   0x7cd49  1      OPC=nop             
  nop                                                 #  179   0x7cd4a  1      OPC=nop             
  nop                                                 #  180   0x7cd4b  1      OPC=nop             
  nop                                                 #  181   0x7cd4c  1      OPC=nop             
  nop                                                 #  182   0x7cd4d  1      OPC=nop             
  nop                                                 #  183   0x7cd4e  1      OPC=nop             
  nop                                                 #  184   0x7cd4f  1      OPC=nop             
  nop                                                 #  185   0x7cd50  1      OPC=nop             
  nop                                                 #  186   0x7cd51  1      OPC=nop             
  nop                                                 #  187   0x7cd52  1      OPC=nop             
  nop                                                 #  188   0x7cd53  1      OPC=nop             
  nop                                                 #  189   0x7cd54  1      OPC=nop             
  nop                                                 #  190   0x7cd55  1      OPC=nop             
  nop                                                 #  191   0x7cd56  1      OPC=nop             
  nop                                                 #  192   0x7cd57  1      OPC=nop             
  nop                                                 #  193   0x7cd58  1      OPC=nop             
  nop                                                 #  194   0x7cd59  1      OPC=nop             
  nop                                                 #  195   0x7cd5a  1      OPC=nop             
  nop                                                 #  196   0x7cd5b  1      OPC=nop             
  nop                                                 #  197   0x7cd5c  1      OPC=nop             
  nop                                                 #  198   0x7cd5d  1      OPC=nop             
  nop                                                 #  199   0x7cd5e  1      OPC=nop             
  nop                                                 #  200   0x7cd5f  1      OPC=nop             
  nop                                                 #  201   0x7cd60  1      OPC=nop             
  callq ._Unwind_Resume                               #  202   0x7cd61  5      OPC=callq_label     
                                                                                                   
.size _ZNSt11__timepunctIcED1Ev, .-_ZNSt11__timepunctIcED1Ev

