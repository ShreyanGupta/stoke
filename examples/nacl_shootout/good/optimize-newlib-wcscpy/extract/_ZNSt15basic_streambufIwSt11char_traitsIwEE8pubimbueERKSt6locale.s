  .text
  .globl _ZNSt15basic_streambufIwSt11char_traitsIwEE8pubimbueERKSt6locale
  .type _ZNSt15basic_streambufIwSt11char_traitsIwEE8pubimbueERKSt6locale, @function

#! file-offset 0xe98a0
#! rip-offset  0xa98a0
#! capacity    256 bytes

# Text                                                              #  Line  RIP      Bytes  Opcode              
._ZNSt15basic_streambufIwSt11char_traitsIwEE8pubimbueERKSt6locale:  #        0xa98a0  0      OPC=<label>         
  movq %r14, -0x8(%rsp)                                             #  1     0xa98a0  5      OPC=movq_m64_r64    
  movl %esi, %r14d                                                  #  2     0xa98a5  3      OPC=movl_r32_r32    
  movq %r13, -0x10(%rsp)                                            #  3     0xa98a8  5      OPC=movq_m64_r64    
  leal 0x1c(%r14), %r13d                                            #  4     0xa98ad  4      OPC=leal_r32_m16    
  movq %rbx, -0x20(%rsp)                                            #  5     0xa98b1  5      OPC=movq_m64_r64    
  movl %edi, %ebx                                                   #  6     0xa98b6  2      OPC=movl_r32_r32    
  movq %r12, -0x18(%rsp)                                            #  7     0xa98b8  5      OPC=movq_m64_r64    
  movl %ebx, %edi                                                   #  8     0xa98bd  2      OPC=movl_r32_r32    
  nop                                                               #  9     0xa98bf  1      OPC=nop             
  subl $0x28, %esp                                                  #  10    0xa98c0  3      OPC=subl_r32_imm8   
  addq %r15, %rsp                                                   #  11    0xa98c3  3      OPC=addq_r64_r64    
  movl %r13d, %esi                                                  #  12    0xa98c6  3      OPC=movl_r32_r32    
  movl %edx, %r12d                                                  #  13    0xa98c9  3      OPC=movl_r32_r32    
  nop                                                               #  14    0xa98cc  1      OPC=nop             
  nop                                                               #  15    0xa98cd  1      OPC=nop             
  nop                                                               #  16    0xa98ce  1      OPC=nop             
  nop                                                               #  17    0xa98cf  1      OPC=nop             
  nop                                                               #  18    0xa98d0  1      OPC=nop             
  nop                                                               #  19    0xa98d1  1      OPC=nop             
  nop                                                               #  20    0xa98d2  1      OPC=nop             
  nop                                                               #  21    0xa98d3  1      OPC=nop             
  nop                                                               #  22    0xa98d4  1      OPC=nop             
  nop                                                               #  23    0xa98d5  1      OPC=nop             
  nop                                                               #  24    0xa98d6  1      OPC=nop             
  nop                                                               #  25    0xa98d7  1      OPC=nop             
  nop                                                               #  26    0xa98d8  1      OPC=nop             
  nop                                                               #  27    0xa98d9  1      OPC=nop             
  nop                                                               #  28    0xa98da  1      OPC=nop             
  callq ._ZNSt6localeC1ERKS_                                        #  29    0xa98db  5      OPC=callq_label     
  movl %r14d, %r14d                                                 #  30    0xa98e0  3      OPC=movl_r32_r32    
  movl (%r15,%r14,1), %eax                                          #  31    0xa98e3  4      OPC=movl_r32_m32    
  movl %r12d, %esi                                                  #  32    0xa98e7  3      OPC=movl_r32_r32    
  movl %r14d, %edi                                                  #  33    0xa98ea  3      OPC=movl_r32_r32    
  movl %eax, %eax                                                   #  34    0xa98ed  2      OPC=movl_r32_r32    
  movl 0x8(%r15,%rax,1), %eax                                       #  35    0xa98ef  5      OPC=movl_r32_m32    
  nop                                                               #  36    0xa98f4  1      OPC=nop             
  nop                                                               #  37    0xa98f5  1      OPC=nop             
  nop                                                               #  38    0xa98f6  1      OPC=nop             
  nop                                                               #  39    0xa98f7  1      OPC=nop             
  andl $0xffffffe0, %eax                                            #  40    0xa98f8  6      OPC=andl_r32_imm32  
  nop                                                               #  41    0xa98fe  1      OPC=nop             
  nop                                                               #  42    0xa98ff  1      OPC=nop             
  nop                                                               #  43    0xa9900  1      OPC=nop             
  addq %r15, %rax                                                   #  44    0xa9901  3      OPC=addq_r64_r64    
  callq %rax                                                        #  45    0xa9904  2      OPC=callq_r64       
  movl %r12d, %esi                                                  #  46    0xa9906  3      OPC=movl_r32_r32    
  movl %r13d, %edi                                                  #  47    0xa9909  3      OPC=movl_r32_r32    
  nop                                                               #  48    0xa990c  1      OPC=nop             
  nop                                                               #  49    0xa990d  1      OPC=nop             
  nop                                                               #  50    0xa990e  1      OPC=nop             
  nop                                                               #  51    0xa990f  1      OPC=nop             
  nop                                                               #  52    0xa9910  1      OPC=nop             
  nop                                                               #  53    0xa9911  1      OPC=nop             
  nop                                                               #  54    0xa9912  1      OPC=nop             
  nop                                                               #  55    0xa9913  1      OPC=nop             
  nop                                                               #  56    0xa9914  1      OPC=nop             
  nop                                                               #  57    0xa9915  1      OPC=nop             
  nop                                                               #  58    0xa9916  1      OPC=nop             
  nop                                                               #  59    0xa9917  1      OPC=nop             
  nop                                                               #  60    0xa9918  1      OPC=nop             
  nop                                                               #  61    0xa9919  1      OPC=nop             
  nop                                                               #  62    0xa991a  1      OPC=nop             
  nop                                                               #  63    0xa991b  1      OPC=nop             
  nop                                                               #  64    0xa991c  1      OPC=nop             
  nop                                                               #  65    0xa991d  1      OPC=nop             
  nop                                                               #  66    0xa991e  1      OPC=nop             
  nop                                                               #  67    0xa991f  1      OPC=nop             
  nop                                                               #  68    0xa9920  1      OPC=nop             
  callq ._ZNSt6localeaSERKS_                                        #  69    0xa9921  5      OPC=callq_label     
  movl %ebx, %eax                                                   #  70    0xa9926  2      OPC=movl_r32_r32    
  movq 0x10(%rsp), %r12                                             #  71    0xa9928  5      OPC=movq_r64_m64    
  movq 0x8(%rsp), %rbx                                              #  72    0xa992d  5      OPC=movq_r64_m64    
  movq 0x18(%rsp), %r13                                             #  73    0xa9932  5      OPC=movq_r64_m64    
  movq 0x20(%rsp), %r14                                             #  74    0xa9937  5      OPC=movq_r64_m64    
  addl $0x28, %esp                                                  #  75    0xa993c  3      OPC=addl_r32_imm8   
  addq %r15, %rsp                                                   #  76    0xa993f  3      OPC=addq_r64_r64    
  popq %r11                                                         #  77    0xa9942  2      OPC=popq_r64_1      
  xchgw %ax, %ax                                                    #  78    0xa9944  2      OPC=xchgw_ax_r16    
  andl $0xffffffe0, %r11d                                           #  79    0xa9946  7      OPC=andl_r32_imm32  
  nop                                                               #  80    0xa994d  1      OPC=nop             
  nop                                                               #  81    0xa994e  1      OPC=nop             
  nop                                                               #  82    0xa994f  1      OPC=nop             
  nop                                                               #  83    0xa9950  1      OPC=nop             
  addq %r15, %r11                                                   #  84    0xa9951  3      OPC=addq_r64_r64    
  jmpq %r11                                                         #  85    0xa9954  3      OPC=jmpq_r64        
  nop                                                               #  86    0xa9957  1      OPC=nop             
  nop                                                               #  87    0xa9958  1      OPC=nop             
  nop                                                               #  88    0xa9959  1      OPC=nop             
  nop                                                               #  89    0xa995a  1      OPC=nop             
  nop                                                               #  90    0xa995b  1      OPC=nop             
  nop                                                               #  91    0xa995c  1      OPC=nop             
  nop                                                               #  92    0xa995d  1      OPC=nop             
  nop                                                               #  93    0xa995e  1      OPC=nop             
  nop                                                               #  94    0xa995f  1      OPC=nop             
  nop                                                               #  95    0xa9960  1      OPC=nop             
  nop                                                               #  96    0xa9961  1      OPC=nop             
  nop                                                               #  97    0xa9962  1      OPC=nop             
  nop                                                               #  98    0xa9963  1      OPC=nop             
  nop                                                               #  99    0xa9964  1      OPC=nop             
  nop                                                               #  100   0xa9965  1      OPC=nop             
  nop                                                               #  101   0xa9966  1      OPC=nop             
  nop                                                               #  102   0xa9967  1      OPC=nop             
  nop                                                               #  103   0xa9968  1      OPC=nop             
  nop                                                               #  104   0xa9969  1      OPC=nop             
  nop                                                               #  105   0xa996a  1      OPC=nop             
  nop                                                               #  106   0xa996b  1      OPC=nop             
  nop                                                               #  107   0xa996c  1      OPC=nop             
  movl %eax, %r12d                                                  #  108   0xa996d  3      OPC=movl_r32_r32    
  movl %ebx, %edi                                                   #  109   0xa9970  2      OPC=movl_r32_r32    
  nop                                                               #  110   0xa9972  1      OPC=nop             
  nop                                                               #  111   0xa9973  1      OPC=nop             
  nop                                                               #  112   0xa9974  1      OPC=nop             
  nop                                                               #  113   0xa9975  1      OPC=nop             
  nop                                                               #  114   0xa9976  1      OPC=nop             
  nop                                                               #  115   0xa9977  1      OPC=nop             
  nop                                                               #  116   0xa9978  1      OPC=nop             
  nop                                                               #  117   0xa9979  1      OPC=nop             
  nop                                                               #  118   0xa997a  1      OPC=nop             
  nop                                                               #  119   0xa997b  1      OPC=nop             
  nop                                                               #  120   0xa997c  1      OPC=nop             
  nop                                                               #  121   0xa997d  1      OPC=nop             
  nop                                                               #  122   0xa997e  1      OPC=nop             
  nop                                                               #  123   0xa997f  1      OPC=nop             
  nop                                                               #  124   0xa9980  1      OPC=nop             
  nop                                                               #  125   0xa9981  1      OPC=nop             
  nop                                                               #  126   0xa9982  1      OPC=nop             
  nop                                                               #  127   0xa9983  1      OPC=nop             
  nop                                                               #  128   0xa9984  1      OPC=nop             
  nop                                                               #  129   0xa9985  1      OPC=nop             
  nop                                                               #  130   0xa9986  1      OPC=nop             
  nop                                                               #  131   0xa9987  1      OPC=nop             
  callq ._ZNSt6localeD1Ev                                           #  132   0xa9988  5      OPC=callq_label     
  movl %r12d, %edi                                                  #  133   0xa998d  3      OPC=movl_r32_r32    
  nop                                                               #  134   0xa9990  1      OPC=nop             
  nop                                                               #  135   0xa9991  1      OPC=nop             
  nop                                                               #  136   0xa9992  1      OPC=nop             
  nop                                                               #  137   0xa9993  1      OPC=nop             
  nop                                                               #  138   0xa9994  1      OPC=nop             
  nop                                                               #  139   0xa9995  1      OPC=nop             
  nop                                                               #  140   0xa9996  1      OPC=nop             
  nop                                                               #  141   0xa9997  1      OPC=nop             
  nop                                                               #  142   0xa9998  1      OPC=nop             
  nop                                                               #  143   0xa9999  1      OPC=nop             
  nop                                                               #  144   0xa999a  1      OPC=nop             
  nop                                                               #  145   0xa999b  1      OPC=nop             
  nop                                                               #  146   0xa999c  1      OPC=nop             
  nop                                                               #  147   0xa999d  1      OPC=nop             
  nop                                                               #  148   0xa999e  1      OPC=nop             
  nop                                                               #  149   0xa999f  1      OPC=nop             
  nop                                                               #  150   0xa99a0  1      OPC=nop             
  nop                                                               #  151   0xa99a1  1      OPC=nop             
  nop                                                               #  152   0xa99a2  1      OPC=nop             
  nop                                                               #  153   0xa99a3  1      OPC=nop             
  nop                                                               #  154   0xa99a4  1      OPC=nop             
  nop                                                               #  155   0xa99a5  1      OPC=nop             
  nop                                                               #  156   0xa99a6  1      OPC=nop             
  nop                                                               #  157   0xa99a7  1      OPC=nop             
  callq ._Unwind_Resume                                             #  158   0xa99a8  5      OPC=callq_label     
                                                                                                                 
.size _ZNSt15basic_streambufIwSt11char_traitsIwEE8pubimbueERKSt6locale, .-_ZNSt15basic_streambufIwSt11char_traitsIwEE8pubimbueERKSt6locale

