  .text
  .globl _ZN5boost11object_poolI4NodeNS_33default_user_allocator_new_deleteEED1Ev
  .type _ZN5boost11object_poolI4NodeNS_33default_user_allocator_new_deleteEED1Ev, @function

#! file-offset 0x626e0
#! rip-offset  0x226e0
#! capacity    320 bytes

# Text                                                                      #  Line  RIP      Bytes  Opcode              
._ZN5boost11object_poolI4NodeNS_33default_user_allocator_new_deleteEED1Ev:  #        0x226e0  0      OPC=<label>         
  pushq %r14                                                                #  1     0x226e0  2      OPC=pushq_r64_1     
  pushq %r13                                                                #  2     0x226e2  2      OPC=pushq_r64_1     
  pushq %r12                                                                #  3     0x226e4  2      OPC=pushq_r64_1     
  pushq %rbx                                                                #  4     0x226e6  1      OPC=pushq_r64_1     
  subl $0x8, %esp                                                           #  5     0x226e7  3      OPC=subl_r32_imm8   
  addq %r15, %rsp                                                           #  6     0x226ea  3      OPC=addq_r64_r64    
  movl %edi, %ebx                                                           #  7     0x226ed  2      OPC=movl_r32_r32    
  movl %ebx, %ebx                                                           #  8     0x226ef  2      OPC=movl_r32_r32    
  movl 0x4(%r15,%rbx,1), %edi                                               #  9     0x226f1  5      OPC=movl_r32_m32    
  testq %rdi, %rdi                                                          #  10    0x226f6  3      OPC=testq_r64_r64   
  je .L_22800                                                               #  11    0x226f9  6      OPC=je_label_1      
  nop                                                                       #  12    0x226ff  1      OPC=nop             
  movl %ebx, %ebx                                                           #  13    0x22700  2      OPC=movl_r32_r32    
  movl 0x8(%r15,%rbx,1), %r8d                                               #  14    0x22702  5      OPC=movl_r32_m32    
  movl %ebx, %ebx                                                           #  15    0x22707  2      OPC=movl_r32_r32    
  movl 0xc(%r15,%rbx,1), %esi                                               #  16    0x22709  5      OPC=movl_r32_m32    
  movl %esi, %eax                                                           #  17    0x2270e  2      OPC=movl_r32_r32    
  movl $0x4, %ecx                                                           #  18    0x22710  5      OPC=movl_r32_imm32  
  nop                                                                       #  19    0x22715  1      OPC=nop             
  nop                                                                       #  20    0x22716  1      OPC=nop             
  nop                                                                       #  21    0x22717  1      OPC=nop             
  nop                                                                       #  22    0x22718  1      OPC=nop             
  nop                                                                       #  23    0x22719  1      OPC=nop             
  nop                                                                       #  24    0x2271a  1      OPC=nop             
  nop                                                                       #  25    0x2271b  1      OPC=nop             
  nop                                                                       #  26    0x2271c  1      OPC=nop             
  nop                                                                       #  27    0x2271d  1      OPC=nop             
  nop                                                                       #  28    0x2271e  1      OPC=nop             
  nop                                                                       #  29    0x2271f  1      OPC=nop             
.L_22720:                                                                   #        0x22720  0      OPC=<label>         
  xorl %edx, %edx                                                           #  30    0x22720  2      OPC=xorl_r32_r32    
  divl %ecx                                                                 #  31    0x22722  2      OPC=divl_r32        
  movl %ecx, %r12d                                                          #  32    0x22724  3      OPC=movl_r32_r32    
  movl %ecx, %eax                                                           #  33    0x22727  2      OPC=movl_r32_r32    
  movl %edx, %ecx                                                           #  34    0x22729  2      OPC=movl_r32_r32    
  testl %edx, %edx                                                          #  35    0x2272b  2      OPC=testl_r32_r32   
  jne .L_22720                                                              #  36    0x2272d  2      OPC=jne_label       
  movl %esi, %eax                                                           #  37    0x2272f  2      OPC=movl_r32_r32    
  xorl %edx, %edx                                                           #  38    0x22731  2      OPC=xorl_r32_r32    
  divl %r12d                                                                #  39    0x22733  3      OPC=divl_r32        
  leal (,%rax,4), %r12d                                                     #  40    0x22736  8      OPC=leal_r32_m16    
  xchgw %ax, %ax                                                            #  41    0x2273e  2      OPC=xchgw_ax_r16    
  movq %rdi, %r13                                                           #  42    0x22740  3      OPC=movq_r64_r64    
  movl %r8d, %r14d                                                          #  43    0x22743  3      OPC=movl_r32_r32    
  nop                                                                       #  44    0x22746  1      OPC=nop             
  nop                                                                       #  45    0x22747  1      OPC=nop             
  nop                                                                       #  46    0x22748  1      OPC=nop             
  nop                                                                       #  47    0x22749  1      OPC=nop             
  nop                                                                       #  48    0x2274a  1      OPC=nop             
  nop                                                                       #  49    0x2274b  1      OPC=nop             
  nop                                                                       #  50    0x2274c  1      OPC=nop             
  nop                                                                       #  51    0x2274d  1      OPC=nop             
  nop                                                                       #  52    0x2274e  1      OPC=nop             
  nop                                                                       #  53    0x2274f  1      OPC=nop             
  nop                                                                       #  54    0x22750  1      OPC=nop             
  nop                                                                       #  55    0x22751  1      OPC=nop             
  nop                                                                       #  56    0x22752  1      OPC=nop             
  nop                                                                       #  57    0x22753  1      OPC=nop             
  nop                                                                       #  58    0x22754  1      OPC=nop             
  nop                                                                       #  59    0x22755  1      OPC=nop             
  nop                                                                       #  60    0x22756  1      OPC=nop             
  nop                                                                       #  61    0x22757  1      OPC=nop             
  nop                                                                       #  62    0x22758  1      OPC=nop             
  nop                                                                       #  63    0x22759  1      OPC=nop             
  nop                                                                       #  64    0x2275a  1      OPC=nop             
  nop                                                                       #  65    0x2275b  1      OPC=nop             
  nop                                                                       #  66    0x2275c  1      OPC=nop             
  nop                                                                       #  67    0x2275d  1      OPC=nop             
  nop                                                                       #  68    0x2275e  1      OPC=nop             
  nop                                                                       #  69    0x2275f  1      OPC=nop             
.L_22760:                                                                   #        0x22760  0      OPC=<label>         
  leal -0x4(%r14,%r13,1), %r8d                                              #  70    0x22760  5      OPC=leal_r32_m16    
  movl %r8d, %r8d                                                           #  71    0x22765  3      OPC=movl_r32_r32    
  movl (%r15,%r8,1), %r14d                                                  #  72    0x22768  4      OPC=movl_r32_m32    
  subl $0x4, %r8d                                                           #  73    0x2276c  4      OPC=subl_r32_imm8   
  movl %r8d, %r8d                                                           #  74    0x22770  3      OPC=movl_r32_r32    
  movl (%r15,%r8,1), %r13d                                                  #  75    0x22773  4      OPC=movl_r32_m32    
  cmpl %edi, %r8d                                                           #  76    0x22777  3      OPC=cmpl_r32_r32    
  je .L_227a0                                                               #  77    0x2277a  2      OPC=je_label        
  leal (%r12,%rdi,1), %eax                                                  #  78    0x2277c  4      OPC=leal_r32_m16    
.L_22780:                                                                   #        0x22780  0      OPC=<label>         
  addl %r12d, %eax                                                          #  79    0x22780  3      OPC=addl_r32_r32    
  movl %eax, %edx                                                           #  80    0x22783  2      OPC=movl_r32_r32    
  subl %r12d, %edx                                                          #  81    0x22785  3      OPC=subl_r32_r32    
  cmpl %r8d, %edx                                                           #  82    0x22788  3      OPC=cmpl_r32_r32    
  jne .L_22780                                                              #  83    0x2278b  2      OPC=jne_label       
  nop                                                                       #  84    0x2278d  1      OPC=nop             
  nop                                                                       #  85    0x2278e  1      OPC=nop             
  nop                                                                       #  86    0x2278f  1      OPC=nop             
  nop                                                                       #  87    0x22790  1      OPC=nop             
  nop                                                                       #  88    0x22791  1      OPC=nop             
  nop                                                                       #  89    0x22792  1      OPC=nop             
  nop                                                                       #  90    0x22793  1      OPC=nop             
  nop                                                                       #  91    0x22794  1      OPC=nop             
  nop                                                                       #  92    0x22795  1      OPC=nop             
  nop                                                                       #  93    0x22796  1      OPC=nop             
  nop                                                                       #  94    0x22797  1      OPC=nop             
  nop                                                                       #  95    0x22798  1      OPC=nop             
  nop                                                                       #  96    0x22799  1      OPC=nop             
  nop                                                                       #  97    0x2279a  1      OPC=nop             
  nop                                                                       #  98    0x2279b  1      OPC=nop             
  nop                                                                       #  99    0x2279c  1      OPC=nop             
  nop                                                                       #  100   0x2279d  1      OPC=nop             
  nop                                                                       #  101   0x2279e  1      OPC=nop             
  nop                                                                       #  102   0x2279f  1      OPC=nop             
.L_227a0:                                                                   #        0x227a0  0      OPC=<label>         
  testq %rdi, %rdi                                                          #  103   0x227a0  3      OPC=testq_r64_r64   
  je .L_227c0                                                               #  104   0x227a3  2      OPC=je_label        
  nop                                                                       #  105   0x227a5  1      OPC=nop             
  nop                                                                       #  106   0x227a6  1      OPC=nop             
  nop                                                                       #  107   0x227a7  1      OPC=nop             
  nop                                                                       #  108   0x227a8  1      OPC=nop             
  nop                                                                       #  109   0x227a9  1      OPC=nop             
  nop                                                                       #  110   0x227aa  1      OPC=nop             
  nop                                                                       #  111   0x227ab  1      OPC=nop             
  nop                                                                       #  112   0x227ac  1      OPC=nop             
  nop                                                                       #  113   0x227ad  1      OPC=nop             
  nop                                                                       #  114   0x227ae  1      OPC=nop             
  nop                                                                       #  115   0x227af  1      OPC=nop             
  nop                                                                       #  116   0x227b0  1      OPC=nop             
  nop                                                                       #  117   0x227b1  1      OPC=nop             
  nop                                                                       #  118   0x227b2  1      OPC=nop             
  nop                                                                       #  119   0x227b3  1      OPC=nop             
  nop                                                                       #  120   0x227b4  1      OPC=nop             
  nop                                                                       #  121   0x227b5  1      OPC=nop             
  nop                                                                       #  122   0x227b6  1      OPC=nop             
  nop                                                                       #  123   0x227b7  1      OPC=nop             
  nop                                                                       #  124   0x227b8  1      OPC=nop             
  nop                                                                       #  125   0x227b9  1      OPC=nop             
  nop                                                                       #  126   0x227ba  1      OPC=nop             
  callq ._ZdaPv                                                             #  127   0x227bb  5      OPC=callq_label     
.L_227c0:                                                                   #        0x227c0  0      OPC=<label>         
  testq %r13, %r13                                                          #  128   0x227c0  3      OPC=testq_r64_r64   
  je .L_227e0                                                               #  129   0x227c3  2      OPC=je_label        
  movq %r13, %rdi                                                           #  130   0x227c5  3      OPC=movq_r64_r64    
  jmpq .L_22760                                                             #  131   0x227c8  2      OPC=jmpq_label      
  nop                                                                       #  132   0x227ca  1      OPC=nop             
  nop                                                                       #  133   0x227cb  1      OPC=nop             
  nop                                                                       #  134   0x227cc  1      OPC=nop             
  nop                                                                       #  135   0x227cd  1      OPC=nop             
  nop                                                                       #  136   0x227ce  1      OPC=nop             
  nop                                                                       #  137   0x227cf  1      OPC=nop             
  nop                                                                       #  138   0x227d0  1      OPC=nop             
  nop                                                                       #  139   0x227d1  1      OPC=nop             
  nop                                                                       #  140   0x227d2  1      OPC=nop             
  nop                                                                       #  141   0x227d3  1      OPC=nop             
  nop                                                                       #  142   0x227d4  1      OPC=nop             
  nop                                                                       #  143   0x227d5  1      OPC=nop             
  nop                                                                       #  144   0x227d6  1      OPC=nop             
  nop                                                                       #  145   0x227d7  1      OPC=nop             
  nop                                                                       #  146   0x227d8  1      OPC=nop             
  nop                                                                       #  147   0x227d9  1      OPC=nop             
  nop                                                                       #  148   0x227da  1      OPC=nop             
  nop                                                                       #  149   0x227db  1      OPC=nop             
  nop                                                                       #  150   0x227dc  1      OPC=nop             
  nop                                                                       #  151   0x227dd  1      OPC=nop             
  nop                                                                       #  152   0x227de  1      OPC=nop             
  nop                                                                       #  153   0x227df  1      OPC=nop             
.L_227e0:                                                                   #        0x227e0  0      OPC=<label>         
  movl %ebx, %ebx                                                           #  154   0x227e0  2      OPC=movl_r32_r32    
  movl $0x0, 0x4(%r15,%rbx,1)                                               #  155   0x227e2  9      OPC=movl_m32_imm32  
  nop                                                                       #  156   0x227eb  1      OPC=nop             
  nop                                                                       #  157   0x227ec  1      OPC=nop             
  nop                                                                       #  158   0x227ed  1      OPC=nop             
  nop                                                                       #  159   0x227ee  1      OPC=nop             
  nop                                                                       #  160   0x227ef  1      OPC=nop             
  nop                                                                       #  161   0x227f0  1      OPC=nop             
  nop                                                                       #  162   0x227f1  1      OPC=nop             
  nop                                                                       #  163   0x227f2  1      OPC=nop             
  nop                                                                       #  164   0x227f3  1      OPC=nop             
  nop                                                                       #  165   0x227f4  1      OPC=nop             
  nop                                                                       #  166   0x227f5  1      OPC=nop             
  nop                                                                       #  167   0x227f6  1      OPC=nop             
  nop                                                                       #  168   0x227f7  1      OPC=nop             
  nop                                                                       #  169   0x227f8  1      OPC=nop             
  nop                                                                       #  170   0x227f9  1      OPC=nop             
  nop                                                                       #  171   0x227fa  1      OPC=nop             
  nop                                                                       #  172   0x227fb  1      OPC=nop             
  nop                                                                       #  173   0x227fc  1      OPC=nop             
  nop                                                                       #  174   0x227fd  1      OPC=nop             
  nop                                                                       #  175   0x227fe  1      OPC=nop             
  nop                                                                       #  176   0x227ff  1      OPC=nop             
.L_22800:                                                                   #        0x22800  0      OPC=<label>         
  addl $0x8, %esp                                                           #  177   0x22800  3      OPC=addl_r32_imm8   
  addq %r15, %rsp                                                           #  178   0x22803  3      OPC=addq_r64_r64    
  popq %rbx                                                                 #  179   0x22806  1      OPC=popq_r64_1      
  popq %r12                                                                 #  180   0x22807  2      OPC=popq_r64_1      
  popq %r13                                                                 #  181   0x22809  2      OPC=popq_r64_1      
  popq %r14                                                                 #  182   0x2280b  2      OPC=popq_r64_1      
  popq %r11                                                                 #  183   0x2280d  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d                                                   #  184   0x2280f  7      OPC=andl_r32_imm32  
  nop                                                                       #  185   0x22816  1      OPC=nop             
  nop                                                                       #  186   0x22817  1      OPC=nop             
  nop                                                                       #  187   0x22818  1      OPC=nop             
  nop                                                                       #  188   0x22819  1      OPC=nop             
  addq %r15, %r11                                                           #  189   0x2281a  3      OPC=addq_r64_r64    
  jmpq %r11                                                                 #  190   0x2281d  3      OPC=jmpq_r64        
  nop                                                                       #  191   0x22820  1      OPC=nop             
  nop                                                                       #  192   0x22821  1      OPC=nop             
  nop                                                                       #  193   0x22822  1      OPC=nop             
  nop                                                                       #  194   0x22823  1      OPC=nop             
  nop                                                                       #  195   0x22824  1      OPC=nop             
  nop                                                                       #  196   0x22825  1      OPC=nop             
  nop                                                                       #  197   0x22826  1      OPC=nop             
                                                                                                                         
.size _ZN5boost11object_poolI4NodeNS_33default_user_allocator_new_deleteEED1Ev, .-_ZN5boost11object_poolI4NodeNS_33default_user_allocator_new_deleteEED1Ev

