  .text
  .globl _ZNSs6appendERKSs
  .type _ZNSs6appendERKSs, @function

#! file-offset 0xec8c0
#! rip-offset  0xac8c0
#! capacity    384 bytes

# Text                          #  Line  RIP      Bytes  Opcode              
._ZNSs6appendERKSs:             #        0xac8c0  0      OPC=<label>         
  movq %r14, -0x8(%rsp)         #  1     0xac8c0  5      OPC=movq_m64_r64    
  movl %esi, %r14d              #  2     0xac8c5  3      OPC=movl_r32_r32    
  movq %rbx, -0x20(%rsp)        #  3     0xac8c8  5      OPC=movq_m64_r64    
  movq %r12, -0x18(%rsp)        #  4     0xac8cd  5      OPC=movq_m64_r64    
  movq %r13, -0x10(%rsp)        #  5     0xac8d2  5      OPC=movq_m64_r64    
  subl $0x28, %esp              #  6     0xac8d7  3      OPC=subl_r32_imm8   
  addq %r15, %rsp               #  7     0xac8da  3      OPC=addq_r64_r64    
  nop                           #  8     0xac8dd  1      OPC=nop             
  nop                           #  9     0xac8de  1      OPC=nop             
  nop                           #  10    0xac8df  1      OPC=nop             
  movl %r14d, %r14d             #  11    0xac8e0  3      OPC=movl_r32_r32    
  movl (%r15,%r14,1), %esi      #  12    0xac8e3  4      OPC=movl_r32_m32    
  movl %edi, %ebx               #  13    0xac8e7  2      OPC=movl_r32_r32    
  leal -0xc(%rsi), %eax         #  14    0xac8e9  3      OPC=leal_r32_m16    
  movl %eax, %eax               #  15    0xac8ec  2      OPC=movl_r32_r32    
  movl (%r15,%rax,1), %r13d     #  16    0xac8ee  4      OPC=movl_r32_m32    
  testl %r13d, %r13d            #  17    0xac8f2  3      OPC=testl_r32_r32   
  je .L_ac9e0                   #  18    0xac8f5  6      OPC=je_label_1      
  nop                           #  19    0xac8fb  1      OPC=nop             
  nop                           #  20    0xac8fc  1      OPC=nop             
  nop                           #  21    0xac8fd  1      OPC=nop             
  nop                           #  22    0xac8fe  1      OPC=nop             
  nop                           #  23    0xac8ff  1      OPC=nop             
  movl %ebx, %ebx               #  24    0xac900  2      OPC=movl_r32_r32    
  movl (%r15,%rbx,1), %eax      #  25    0xac902  4      OPC=movl_r32_m32    
  leal -0xc(%rax), %edx         #  26    0xac906  3      OPC=leal_r32_m16    
  movl %edx, %edx               #  27    0xac909  2      OPC=movl_r32_r32    
  movl (%r15,%rdx,1), %edi      #  28    0xac90b  4      OPC=movl_r32_m32    
  leal (%rdi,%r13,1), %r12d     #  29    0xac90f  4      OPC=leal_r32_m16    
  movl %edx, %edx               #  30    0xac913  2      OPC=movl_r32_r32    
  cmpl 0x4(%r15,%rdx,1), %r12d  #  31    0xac915  5      OPC=cmpl_r32_m32    
  ja .L_ac940                   #  32    0xac91a  2      OPC=ja_label        
  nop                           #  33    0xac91c  1      OPC=nop             
  nop                           #  34    0xac91d  1      OPC=nop             
  nop                           #  35    0xac91e  1      OPC=nop             
  nop                           #  36    0xac91f  1      OPC=nop             
  movl %edx, %edx               #  37    0xac920  2      OPC=movl_r32_r32    
  movl 0x8(%r15,%rdx,1), %r9d   #  38    0xac922  5      OPC=movl_r32_m32    
  testl %r9d, %r9d              #  39    0xac927  3      OPC=testl_r32_r32   
  jle .L_ac980                  #  40    0xac92a  2      OPC=jle_label       
  nop                           #  41    0xac92c  1      OPC=nop             
  nop                           #  42    0xac92d  1      OPC=nop             
  nop                           #  43    0xac92e  1      OPC=nop             
  nop                           #  44    0xac92f  1      OPC=nop             
  nop                           #  45    0xac930  1      OPC=nop             
  nop                           #  46    0xac931  1      OPC=nop             
  nop                           #  47    0xac932  1      OPC=nop             
  nop                           #  48    0xac933  1      OPC=nop             
  nop                           #  49    0xac934  1      OPC=nop             
  nop                           #  50    0xac935  1      OPC=nop             
  nop                           #  51    0xac936  1      OPC=nop             
  nop                           #  52    0xac937  1      OPC=nop             
  nop                           #  53    0xac938  1      OPC=nop             
  nop                           #  54    0xac939  1      OPC=nop             
  nop                           #  55    0xac93a  1      OPC=nop             
  nop                           #  56    0xac93b  1      OPC=nop             
  nop                           #  57    0xac93c  1      OPC=nop             
  nop                           #  58    0xac93d  1      OPC=nop             
  nop                           #  59    0xac93e  1      OPC=nop             
  nop                           #  60    0xac93f  1      OPC=nop             
.L_ac940:                       #        0xac940  0      OPC=<label>         
  movl %r12d, %esi              #  61    0xac940  3      OPC=movl_r32_r32    
  movl %ebx, %edi               #  62    0xac943  2      OPC=movl_r32_r32    
  nop                           #  63    0xac945  1      OPC=nop             
  nop                           #  64    0xac946  1      OPC=nop             
  nop                           #  65    0xac947  1      OPC=nop             
  nop                           #  66    0xac948  1      OPC=nop             
  nop                           #  67    0xac949  1      OPC=nop             
  nop                           #  68    0xac94a  1      OPC=nop             
  nop                           #  69    0xac94b  1      OPC=nop             
  nop                           #  70    0xac94c  1      OPC=nop             
  nop                           #  71    0xac94d  1      OPC=nop             
  nop                           #  72    0xac94e  1      OPC=nop             
  nop                           #  73    0xac94f  1      OPC=nop             
  nop                           #  74    0xac950  1      OPC=nop             
  nop                           #  75    0xac951  1      OPC=nop             
  nop                           #  76    0xac952  1      OPC=nop             
  nop                           #  77    0xac953  1      OPC=nop             
  nop                           #  78    0xac954  1      OPC=nop             
  nop                           #  79    0xac955  1      OPC=nop             
  nop                           #  80    0xac956  1      OPC=nop             
  nop                           #  81    0xac957  1      OPC=nop             
  nop                           #  82    0xac958  1      OPC=nop             
  nop                           #  83    0xac959  1      OPC=nop             
  nop                           #  84    0xac95a  1      OPC=nop             
  callq ._ZNSs7reserveEj        #  85    0xac95b  5      OPC=callq_label     
  movl %ebx, %ebx               #  86    0xac960  2      OPC=movl_r32_r32    
  movl (%r15,%rbx,1), %eax      #  87    0xac962  4      OPC=movl_r32_m32    
  movl %r14d, %r14d             #  88    0xac966  3      OPC=movl_r32_r32    
  movl (%r15,%r14,1), %esi      #  89    0xac969  4      OPC=movl_r32_m32    
  leal -0xc(%rax), %edx         #  90    0xac96d  3      OPC=leal_r32_m16    
  movl %edx, %edx               #  91    0xac970  2      OPC=movl_r32_r32    
  movl (%r15,%rdx,1), %edi      #  92    0xac972  4      OPC=movl_r32_m32    
  nop                           #  93    0xac976  1      OPC=nop             
  nop                           #  94    0xac977  1      OPC=nop             
  nop                           #  95    0xac978  1      OPC=nop             
  nop                           #  96    0xac979  1      OPC=nop             
  nop                           #  97    0xac97a  1      OPC=nop             
  nop                           #  98    0xac97b  1      OPC=nop             
  nop                           #  99    0xac97c  1      OPC=nop             
  nop                           #  100   0xac97d  1      OPC=nop             
  nop                           #  101   0xac97e  1      OPC=nop             
  nop                           #  102   0xac97f  1      OPC=nop             
.L_ac980:                       #        0xac980  0      OPC=<label>         
  addl %eax, %edi               #  103   0xac980  2      OPC=addl_r32_r32    
  cmpl $0x1, %r13d              #  104   0xac982  4      OPC=cmpl_r32_imm8   
  je .L_aca20                   #  105   0xac986  6      OPC=je_label_1      
  movl %r13d, %edx              #  106   0xac98c  3      OPC=movl_r32_r32    
  nop                           #  107   0xac98f  1      OPC=nop             
  nop                           #  108   0xac990  1      OPC=nop             
  nop                           #  109   0xac991  1      OPC=nop             
  nop                           #  110   0xac992  1      OPC=nop             
  nop                           #  111   0xac993  1      OPC=nop             
  nop                           #  112   0xac994  1      OPC=nop             
  nop                           #  113   0xac995  1      OPC=nop             
  nop                           #  114   0xac996  1      OPC=nop             
  nop                           #  115   0xac997  1      OPC=nop             
  nop                           #  116   0xac998  1      OPC=nop             
  nop                           #  117   0xac999  1      OPC=nop             
  nop                           #  118   0xac99a  1      OPC=nop             
  callq .memcpy                 #  119   0xac99b  5      OPC=callq_label     
.L_ac9a0:                       #        0xac9a0  0      OPC=<label>         
  movl %ebx, %ebx               #  120   0xac9a0  2      OPC=movl_r32_r32    
  movl (%r15,%rbx,1), %eax      #  121   0xac9a2  4      OPC=movl_r32_m32    
  subl $0xc, %eax               #  122   0xac9a6  3      OPC=subl_r32_imm8   
  movl %eax, %eax               #  123   0xac9a9  2      OPC=movl_r32_r32    
  movl %r12d, (%r15,%rax,1)     #  124   0xac9ab  4      OPC=movl_m32_r32    
  addl %eax, %r12d              #  125   0xac9af  3      OPC=addl_r32_r32    
  movl %eax, %eax               #  126   0xac9b2  2      OPC=movl_r32_r32    
  movl $0x0, 0x8(%r15,%rax,1)   #  127   0xac9b4  9      OPC=movl_m32_imm32  
  nop                           #  128   0xac9bd  1      OPC=nop             
  nop                           #  129   0xac9be  1      OPC=nop             
  nop                           #  130   0xac9bf  1      OPC=nop             
  movl %r12d, %r12d             #  131   0xac9c0  3      OPC=movl_r32_r32    
  movb $0x0, 0xc(%r15,%r12,1)   #  132   0xac9c3  6      OPC=movb_m8_imm8    
  nop                           #  133   0xac9c9  1      OPC=nop             
  nop                           #  134   0xac9ca  1      OPC=nop             
  nop                           #  135   0xac9cb  1      OPC=nop             
  nop                           #  136   0xac9cc  1      OPC=nop             
  nop                           #  137   0xac9cd  1      OPC=nop             
  nop                           #  138   0xac9ce  1      OPC=nop             
  nop                           #  139   0xac9cf  1      OPC=nop             
  nop                           #  140   0xac9d0  1      OPC=nop             
  nop                           #  141   0xac9d1  1      OPC=nop             
  nop                           #  142   0xac9d2  1      OPC=nop             
  nop                           #  143   0xac9d3  1      OPC=nop             
  nop                           #  144   0xac9d4  1      OPC=nop             
  nop                           #  145   0xac9d5  1      OPC=nop             
  nop                           #  146   0xac9d6  1      OPC=nop             
  nop                           #  147   0xac9d7  1      OPC=nop             
  nop                           #  148   0xac9d8  1      OPC=nop             
  nop                           #  149   0xac9d9  1      OPC=nop             
  nop                           #  150   0xac9da  1      OPC=nop             
  nop                           #  151   0xac9db  1      OPC=nop             
  nop                           #  152   0xac9dc  1      OPC=nop             
  nop                           #  153   0xac9dd  1      OPC=nop             
  nop                           #  154   0xac9de  1      OPC=nop             
  nop                           #  155   0xac9df  1      OPC=nop             
.L_ac9e0:                       #        0xac9e0  0      OPC=<label>         
  movl %ebx, %eax               #  156   0xac9e0  2      OPC=movl_r32_r32    
  movq 0x10(%rsp), %r12         #  157   0xac9e2  5      OPC=movq_r64_m64    
  movq 0x8(%rsp), %rbx          #  158   0xac9e7  5      OPC=movq_r64_m64    
  movq 0x18(%rsp), %r13         #  159   0xac9ec  5      OPC=movq_r64_m64    
  movq 0x20(%rsp), %r14         #  160   0xac9f1  5      OPC=movq_r64_m64    
  addl $0x28, %esp              #  161   0xac9f6  3      OPC=addl_r32_imm8   
  addq %r15, %rsp               #  162   0xac9f9  3      OPC=addq_r64_r64    
  popq %r11                     #  163   0xac9fc  2      OPC=popq_r64_1      
  xchgw %ax, %ax                #  164   0xac9fe  2      OPC=xchgw_ax_r16    
  andl $0xffffffe0, %r11d       #  165   0xaca00  7      OPC=andl_r32_imm32  
  nop                           #  166   0xaca07  1      OPC=nop             
  nop                           #  167   0xaca08  1      OPC=nop             
  nop                           #  168   0xaca09  1      OPC=nop             
  nop                           #  169   0xaca0a  1      OPC=nop             
  addq %r15, %r11               #  170   0xaca0b  3      OPC=addq_r64_r64    
  jmpq %r11                     #  171   0xaca0e  3      OPC=jmpq_r64        
  nop                           #  172   0xaca11  1      OPC=nop             
  nop                           #  173   0xaca12  1      OPC=nop             
  nop                           #  174   0xaca13  1      OPC=nop             
  nop                           #  175   0xaca14  1      OPC=nop             
  nop                           #  176   0xaca15  1      OPC=nop             
  nop                           #  177   0xaca16  1      OPC=nop             
  nop                           #  178   0xaca17  1      OPC=nop             
  nop                           #  179   0xaca18  1      OPC=nop             
  nop                           #  180   0xaca19  1      OPC=nop             
  nop                           #  181   0xaca1a  1      OPC=nop             
  nop                           #  182   0xaca1b  1      OPC=nop             
  nop                           #  183   0xaca1c  1      OPC=nop             
  nop                           #  184   0xaca1d  1      OPC=nop             
  nop                           #  185   0xaca1e  1      OPC=nop             
  nop                           #  186   0xaca1f  1      OPC=nop             
  nop                           #  187   0xaca20  1      OPC=nop             
  nop                           #  188   0xaca21  1      OPC=nop             
  nop                           #  189   0xaca22  1      OPC=nop             
  nop                           #  190   0xaca23  1      OPC=nop             
  nop                           #  191   0xaca24  1      OPC=nop             
  nop                           #  192   0xaca25  1      OPC=nop             
  nop                           #  193   0xaca26  1      OPC=nop             
.L_aca20:                       #        0xaca27  0      OPC=<label>         
  movl %esi, %esi               #  194   0xaca27  2      OPC=movl_r32_r32    
  movzbl (%r15,%rsi,1), %eax    #  195   0xaca29  5      OPC=movzbl_r32_m8   
  movl %edi, %edi               #  196   0xaca2e  2      OPC=movl_r32_r32    
  movb %al, (%r15,%rdi,1)       #  197   0xaca30  4      OPC=movb_m8_r8      
  jmpq .L_ac9a0                 #  198   0xaca34  5      OPC=jmpq_label_1    
  nop                           #  199   0xaca39  1      OPC=nop             
  nop                           #  200   0xaca3a  1      OPC=nop             
  nop                           #  201   0xaca3b  1      OPC=nop             
  nop                           #  202   0xaca3c  1      OPC=nop             
  nop                           #  203   0xaca3d  1      OPC=nop             
  nop                           #  204   0xaca3e  1      OPC=nop             
  nop                           #  205   0xaca3f  1      OPC=nop             
  nop                           #  206   0xaca40  1      OPC=nop             
  nop                           #  207   0xaca41  1      OPC=nop             
  nop                           #  208   0xaca42  1      OPC=nop             
  nop                           #  209   0xaca43  1      OPC=nop             
  nop                           #  210   0xaca44  1      OPC=nop             
  nop                           #  211   0xaca45  1      OPC=nop             
  nop                           #  212   0xaca46  1      OPC=nop             
                                                                             
.size _ZNSs6appendERKSs, .-_ZNSs6appendERKSs

