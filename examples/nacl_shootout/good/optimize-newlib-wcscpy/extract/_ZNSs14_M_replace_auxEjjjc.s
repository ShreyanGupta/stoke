  .text
  .globl _ZNSs14_M_replace_auxEjjjc
  .type _ZNSs14_M_replace_auxEjjjc, @function

#! file-offset 0xee500
#! rip-offset  0xae500
#! capacity    320 bytes

# Text                                  #  Line  RIP      Bytes  Opcode              
._ZNSs14_M_replace_auxEjjjc:            #        0xae500  0      OPC=<label>         
  movq %r12, -0x18(%rsp)                #  1     0xae500  5      OPC=movq_m64_r64    
  movl %edi, %r12d                      #  2     0xae505  3      OPC=movl_r32_r32    
  movq %rbx, -0x20(%rsp)                #  3     0xae508  5      OPC=movq_m64_r64    
  movq %r13, -0x10(%rsp)                #  4     0xae50d  5      OPC=movq_m64_r64    
  movq %r14, -0x8(%rsp)                 #  5     0xae512  5      OPC=movq_m64_r64    
  subl $0x38, %esp                      #  6     0xae517  3      OPC=subl_r32_imm8   
  addq %r15, %rsp                       #  7     0xae51a  3      OPC=addq_r64_r64    
  nop                                   #  8     0xae51d  1      OPC=nop             
  nop                                   #  9     0xae51e  1      OPC=nop             
  nop                                   #  10    0xae51f  1      OPC=nop             
  movl %r12d, %r12d                     #  11    0xae520  3      OPC=movl_r32_r32    
  movl (%r15,%r12,1), %eax              #  12    0xae523  4      OPC=movl_r32_m32    
  movl %ecx, %ebx                       #  13    0xae527  2      OPC=movl_r32_r32    
  movl %edx, %ecx                       #  14    0xae529  2      OPC=movl_r32_r32    
  movl %esi, %r13d                      #  15    0xae52b  3      OPC=movl_r32_r32    
  movl %r8d, %r14d                      #  16    0xae52e  3      OPC=movl_r32_r32    
  movb %r8b, 0xf(%rsp)                  #  17    0xae531  5      OPC=movb_m8_r8      
  subl $0xc, %eax                       #  18    0xae536  3      OPC=subl_r32_imm8   
  movl %eax, %eax                       #  19    0xae539  2      OPC=movl_r32_r32    
  subl (%r15,%rax,1), %ecx              #  20    0xae53b  4      OPC=subl_r32_m32    
  nop                                   #  21    0xae53f  1      OPC=nop             
  movl %ecx, %eax                       #  22    0xae540  2      OPC=movl_r32_r32    
  addl $0x3ffffffc, %eax                #  23    0xae542  5      OPC=addl_eax_imm32  
  cmpl %eax, %ebx                       #  24    0xae547  2      OPC=cmpl_r32_r32    
  ja .L_ae620                           #  25    0xae549  6      OPC=ja_label_1      
  movl %ebx, %ecx                       #  26    0xae54f  2      OPC=movl_r32_r32    
  movl %r12d, %edi                      #  27    0xae551  3      OPC=movl_r32_r32    
  nop                                   #  28    0xae554  1      OPC=nop             
  nop                                   #  29    0xae555  1      OPC=nop             
  nop                                   #  30    0xae556  1      OPC=nop             
  nop                                   #  31    0xae557  1      OPC=nop             
  nop                                   #  32    0xae558  1      OPC=nop             
  nop                                   #  33    0xae559  1      OPC=nop             
  nop                                   #  34    0xae55a  1      OPC=nop             
  callq ._ZNSs9_M_mutateEjjj            #  35    0xae55b  5      OPC=callq_label     
  testl %ebx, %ebx                      #  36    0xae560  2      OPC=testl_r32_r32   
  jne .L_ae5c0                          #  37    0xae562  2      OPC=jne_label       
  nop                                   #  38    0xae564  1      OPC=nop             
  nop                                   #  39    0xae565  1      OPC=nop             
  nop                                   #  40    0xae566  1      OPC=nop             
  nop                                   #  41    0xae567  1      OPC=nop             
  nop                                   #  42    0xae568  1      OPC=nop             
  nop                                   #  43    0xae569  1      OPC=nop             
  nop                                   #  44    0xae56a  1      OPC=nop             
  nop                                   #  45    0xae56b  1      OPC=nop             
  nop                                   #  46    0xae56c  1      OPC=nop             
  nop                                   #  47    0xae56d  1      OPC=nop             
  nop                                   #  48    0xae56e  1      OPC=nop             
  nop                                   #  49    0xae56f  1      OPC=nop             
  nop                                   #  50    0xae570  1      OPC=nop             
  nop                                   #  51    0xae571  1      OPC=nop             
  nop                                   #  52    0xae572  1      OPC=nop             
  nop                                   #  53    0xae573  1      OPC=nop             
  nop                                   #  54    0xae574  1      OPC=nop             
  nop                                   #  55    0xae575  1      OPC=nop             
  nop                                   #  56    0xae576  1      OPC=nop             
  nop                                   #  57    0xae577  1      OPC=nop             
  nop                                   #  58    0xae578  1      OPC=nop             
  nop                                   #  59    0xae579  1      OPC=nop             
  nop                                   #  60    0xae57a  1      OPC=nop             
  nop                                   #  61    0xae57b  1      OPC=nop             
  nop                                   #  62    0xae57c  1      OPC=nop             
  nop                                   #  63    0xae57d  1      OPC=nop             
  nop                                   #  64    0xae57e  1      OPC=nop             
  nop                                   #  65    0xae57f  1      OPC=nop             
.L_ae580:                               #        0xae580  0      OPC=<label>         
  movl %r12d, %eax                      #  66    0xae580  3      OPC=movl_r32_r32    
  movq 0x18(%rsp), %rbx                 #  67    0xae583  5      OPC=movq_r64_m64    
  movq 0x20(%rsp), %r12                 #  68    0xae588  5      OPC=movq_r64_m64    
  movq 0x28(%rsp), %r13                 #  69    0xae58d  5      OPC=movq_r64_m64    
  movq 0x30(%rsp), %r14                 #  70    0xae592  5      OPC=movq_r64_m64    
  addl $0x38, %esp                      #  71    0xae597  3      OPC=addl_r32_imm8   
  addq %r15, %rsp                       #  72    0xae59a  3      OPC=addq_r64_r64    
  popq %r11                             #  73    0xae59d  2      OPC=popq_r64_1      
  nop                                   #  74    0xae59f  1      OPC=nop             
  andl $0xffffffe0, %r11d               #  75    0xae5a0  7      OPC=andl_r32_imm32  
  nop                                   #  76    0xae5a7  1      OPC=nop             
  nop                                   #  77    0xae5a8  1      OPC=nop             
  nop                                   #  78    0xae5a9  1      OPC=nop             
  nop                                   #  79    0xae5aa  1      OPC=nop             
  addq %r15, %r11                       #  80    0xae5ab  3      OPC=addq_r64_r64    
  jmpq %r11                             #  81    0xae5ae  3      OPC=jmpq_r64        
  nop                                   #  82    0xae5b1  1      OPC=nop             
  nop                                   #  83    0xae5b2  1      OPC=nop             
  nop                                   #  84    0xae5b3  1      OPC=nop             
  nop                                   #  85    0xae5b4  1      OPC=nop             
  nop                                   #  86    0xae5b5  1      OPC=nop             
  nop                                   #  87    0xae5b6  1      OPC=nop             
  nop                                   #  88    0xae5b7  1      OPC=nop             
  nop                                   #  89    0xae5b8  1      OPC=nop             
  nop                                   #  90    0xae5b9  1      OPC=nop             
  nop                                   #  91    0xae5ba  1      OPC=nop             
  nop                                   #  92    0xae5bb  1      OPC=nop             
  nop                                   #  93    0xae5bc  1      OPC=nop             
  nop                                   #  94    0xae5bd  1      OPC=nop             
  nop                                   #  95    0xae5be  1      OPC=nop             
  nop                                   #  96    0xae5bf  1      OPC=nop             
  nop                                   #  97    0xae5c0  1      OPC=nop             
  nop                                   #  98    0xae5c1  1      OPC=nop             
  nop                                   #  99    0xae5c2  1      OPC=nop             
  nop                                   #  100   0xae5c3  1      OPC=nop             
  nop                                   #  101   0xae5c4  1      OPC=nop             
  nop                                   #  102   0xae5c5  1      OPC=nop             
  nop                                   #  103   0xae5c6  1      OPC=nop             
.L_ae5c0:                               #        0xae5c7  0      OPC=<label>         
  movl %r12d, %r12d                     #  104   0xae5c7  3      OPC=movl_r32_r32    
  movl (%r15,%r12,1), %edi              #  105   0xae5ca  4      OPC=movl_r32_m32    
  addl %r13d, %edi                      #  106   0xae5ce  3      OPC=addl_r32_r32    
  cmpl $0x1, %ebx                       #  107   0xae5d1  3      OPC=cmpl_r32_imm8   
  je .L_ae600                           #  108   0xae5d4  2      OPC=je_label        
  movsbl 0xf(%rsp), %esi                #  109   0xae5d6  5      OPC=movsbl_r32_m8   
  movl %ebx, %edx                       #  110   0xae5db  2      OPC=movl_r32_r32    
  nop                                   #  111   0xae5dd  1      OPC=nop             
  nop                                   #  112   0xae5de  1      OPC=nop             
  nop                                   #  113   0xae5df  1      OPC=nop             
  nop                                   #  114   0xae5e0  1      OPC=nop             
  nop                                   #  115   0xae5e1  1      OPC=nop             
  callq .memset                         #  116   0xae5e2  5      OPC=callq_label     
  jmpq .L_ae580                         #  117   0xae5e7  2      OPC=jmpq_label      
  nop                                   #  118   0xae5e9  1      OPC=nop             
  nop                                   #  119   0xae5ea  1      OPC=nop             
  nop                                   #  120   0xae5eb  1      OPC=nop             
  nop                                   #  121   0xae5ec  1      OPC=nop             
  nop                                   #  122   0xae5ed  1      OPC=nop             
  nop                                   #  123   0xae5ee  1      OPC=nop             
  nop                                   #  124   0xae5ef  1      OPC=nop             
  nop                                   #  125   0xae5f0  1      OPC=nop             
  nop                                   #  126   0xae5f1  1      OPC=nop             
  nop                                   #  127   0xae5f2  1      OPC=nop             
  nop                                   #  128   0xae5f3  1      OPC=nop             
  nop                                   #  129   0xae5f4  1      OPC=nop             
  nop                                   #  130   0xae5f5  1      OPC=nop             
  nop                                   #  131   0xae5f6  1      OPC=nop             
  nop                                   #  132   0xae5f7  1      OPC=nop             
  nop                                   #  133   0xae5f8  1      OPC=nop             
  nop                                   #  134   0xae5f9  1      OPC=nop             
  nop                                   #  135   0xae5fa  1      OPC=nop             
  nop                                   #  136   0xae5fb  1      OPC=nop             
  nop                                   #  137   0xae5fc  1      OPC=nop             
  nop                                   #  138   0xae5fd  1      OPC=nop             
  nop                                   #  139   0xae5fe  1      OPC=nop             
  nop                                   #  140   0xae5ff  1      OPC=nop             
  nop                                   #  141   0xae600  1      OPC=nop             
  nop                                   #  142   0xae601  1      OPC=nop             
  nop                                   #  143   0xae602  1      OPC=nop             
  nop                                   #  144   0xae603  1      OPC=nop             
  nop                                   #  145   0xae604  1      OPC=nop             
  nop                                   #  146   0xae605  1      OPC=nop             
  nop                                   #  147   0xae606  1      OPC=nop             
.L_ae600:                               #        0xae607  0      OPC=<label>         
  movl %edi, %edi                       #  148   0xae607  2      OPC=movl_r32_r32    
  movb %r14b, (%r15,%rdi,1)             #  149   0xae609  4      OPC=movb_m8_r8      
  jmpq .L_ae580                         #  150   0xae60d  5      OPC=jmpq_label_1    
  nop                                   #  151   0xae612  1      OPC=nop             
  nop                                   #  152   0xae613  1      OPC=nop             
  nop                                   #  153   0xae614  1      OPC=nop             
  nop                                   #  154   0xae615  1      OPC=nop             
  nop                                   #  155   0xae616  1      OPC=nop             
  nop                                   #  156   0xae617  1      OPC=nop             
  nop                                   #  157   0xae618  1      OPC=nop             
  nop                                   #  158   0xae619  1      OPC=nop             
  nop                                   #  159   0xae61a  1      OPC=nop             
  nop                                   #  160   0xae61b  1      OPC=nop             
  nop                                   #  161   0xae61c  1      OPC=nop             
  nop                                   #  162   0xae61d  1      OPC=nop             
  nop                                   #  163   0xae61e  1      OPC=nop             
  nop                                   #  164   0xae61f  1      OPC=nop             
  nop                                   #  165   0xae620  1      OPC=nop             
  nop                                   #  166   0xae621  1      OPC=nop             
  nop                                   #  167   0xae622  1      OPC=nop             
  nop                                   #  168   0xae623  1      OPC=nop             
  nop                                   #  169   0xae624  1      OPC=nop             
  nop                                   #  170   0xae625  1      OPC=nop             
  nop                                   #  171   0xae626  1      OPC=nop             
.L_ae620:                               #        0xae627  0      OPC=<label>         
  movl $0x1003bb40, %edi                #  172   0xae627  5      OPC=movl_r32_imm32  
  nop                                   #  173   0xae62c  1      OPC=nop             
  nop                                   #  174   0xae62d  1      OPC=nop             
  nop                                   #  175   0xae62e  1      OPC=nop             
  nop                                   #  176   0xae62f  1      OPC=nop             
  nop                                   #  177   0xae630  1      OPC=nop             
  nop                                   #  178   0xae631  1      OPC=nop             
  nop                                   #  179   0xae632  1      OPC=nop             
  nop                                   #  180   0xae633  1      OPC=nop             
  nop                                   #  181   0xae634  1      OPC=nop             
  nop                                   #  182   0xae635  1      OPC=nop             
  nop                                   #  183   0xae636  1      OPC=nop             
  nop                                   #  184   0xae637  1      OPC=nop             
  nop                                   #  185   0xae638  1      OPC=nop             
  nop                                   #  186   0xae639  1      OPC=nop             
  nop                                   #  187   0xae63a  1      OPC=nop             
  nop                                   #  188   0xae63b  1      OPC=nop             
  nop                                   #  189   0xae63c  1      OPC=nop             
  nop                                   #  190   0xae63d  1      OPC=nop             
  nop                                   #  191   0xae63e  1      OPC=nop             
  nop                                   #  192   0xae63f  1      OPC=nop             
  nop                                   #  193   0xae640  1      OPC=nop             
  nop                                   #  194   0xae641  1      OPC=nop             
  callq ._ZSt20__throw_length_errorPKc  #  195   0xae642  5      OPC=callq_label     
                                                                                     
.size _ZNSs14_M_replace_auxEjjjc, .-_ZNSs14_M_replace_auxEjjjc

