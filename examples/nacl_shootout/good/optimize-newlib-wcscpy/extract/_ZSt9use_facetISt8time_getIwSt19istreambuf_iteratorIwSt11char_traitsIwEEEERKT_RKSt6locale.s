  .text
  .globl _ZSt9use_facetISt8time_getIwSt19istreambuf_iteratorIwSt11char_traitsIwEEEERKT_RKSt6locale
  .type _ZSt9use_facetISt8time_getIwSt19istreambuf_iteratorIwSt11char_traitsIwEEEERKT_RKSt6locale, @function

#! file-offset 0xf3980
#! rip-offset  0xb3980
#! capacity    192 bytes

# Text                                                                                       #  Line  RIP      Bytes  Opcode              
._ZSt9use_facetISt8time_getIwSt19istreambuf_iteratorIwSt11char_traitsIwEEEERKT_RKSt6locale:  #        0xb3980  0      OPC=<label>         
  pushq %rbx                                                                                 #  1     0xb3980  1      OPC=pushq_r64_1     
  movl %edi, %ebx                                                                            #  2     0xb3981  2      OPC=movl_r32_r32    
  movl $0x100735b4, %edi                                                                     #  3     0xb3983  5      OPC=movl_r32_imm32  
  nop                                                                                        #  4     0xb3988  1      OPC=nop             
  nop                                                                                        #  5     0xb3989  1      OPC=nop             
  nop                                                                                        #  6     0xb398a  1      OPC=nop             
  nop                                                                                        #  7     0xb398b  1      OPC=nop             
  nop                                                                                        #  8     0xb398c  1      OPC=nop             
  nop                                                                                        #  9     0xb398d  1      OPC=nop             
  nop                                                                                        #  10    0xb398e  1      OPC=nop             
  nop                                                                                        #  11    0xb398f  1      OPC=nop             
  nop                                                                                        #  12    0xb3990  1      OPC=nop             
  nop                                                                                        #  13    0xb3991  1      OPC=nop             
  nop                                                                                        #  14    0xb3992  1      OPC=nop             
  nop                                                                                        #  15    0xb3993  1      OPC=nop             
  nop                                                                                        #  16    0xb3994  1      OPC=nop             
  nop                                                                                        #  17    0xb3995  1      OPC=nop             
  nop                                                                                        #  18    0xb3996  1      OPC=nop             
  nop                                                                                        #  19    0xb3997  1      OPC=nop             
  nop                                                                                        #  20    0xb3998  1      OPC=nop             
  nop                                                                                        #  21    0xb3999  1      OPC=nop             
  nop                                                                                        #  22    0xb399a  1      OPC=nop             
  callq ._ZNKSt6locale2id5_M_idEv                                                            #  23    0xb399b  5      OPC=callq_label     
  movl %ebx, %ebx                                                                            #  24    0xb39a0  2      OPC=movl_r32_r32    
  movl (%r15,%rbx,1), %edx                                                                   #  25    0xb39a2  4      OPC=movl_r32_m32    
  movl %edx, %edx                                                                            #  26    0xb39a6  2      OPC=movl_r32_r32    
  cmpl 0x8(%r15,%rdx,1), %eax                                                                #  27    0xb39a8  5      OPC=cmpl_r32_m32    
  movl %edx, %edx                                                                            #  28    0xb39ad  2      OPC=movl_r32_r32    
  movl 0x4(%r15,%rdx,1), %ecx                                                                #  29    0xb39af  5      OPC=movl_r32_m32    
  jae .L_b3a00                                                                               #  30    0xb39b4  2      OPC=jae_label       
  leal (%rcx,%rax,4), %eax                                                                   #  31    0xb39b6  3      OPC=leal_r32_m16    
  movl %eax, %eax                                                                            #  32    0xb39b9  2      OPC=movl_r32_r32    
  movl (%r15,%rax,1), %edi                                                                   #  33    0xb39bb  4      OPC=movl_r32_m32    
  nop                                                                                        #  34    0xb39bf  1      OPC=nop             
  testq %rdi, %rdi                                                                           #  35    0xb39c0  3      OPC=testq_r64_r64   
  je .L_b3a00                                                                                #  36    0xb39c3  2      OPC=je_label        
  xorl %ecx, %ecx                                                                            #  37    0xb39c5  2      OPC=xorl_r32_r32    
  movl $0x1003c440, %edx                                                                     #  38    0xb39c7  5      OPC=movl_r32_imm32  
  movl $0x1003a2f4, %esi                                                                     #  39    0xb39cc  5      OPC=movl_r32_imm32  
  nop                                                                                        #  40    0xb39d1  1      OPC=nop             
  nop                                                                                        #  41    0xb39d2  1      OPC=nop             
  nop                                                                                        #  42    0xb39d3  1      OPC=nop             
  nop                                                                                        #  43    0xb39d4  1      OPC=nop             
  nop                                                                                        #  44    0xb39d5  1      OPC=nop             
  nop                                                                                        #  45    0xb39d6  1      OPC=nop             
  nop                                                                                        #  46    0xb39d7  1      OPC=nop             
  nop                                                                                        #  47    0xb39d8  1      OPC=nop             
  nop                                                                                        #  48    0xb39d9  1      OPC=nop             
  nop                                                                                        #  49    0xb39da  1      OPC=nop             
  callq .__dynamic_cast                                                                      #  50    0xb39db  5      OPC=callq_label     
  movl %eax, %eax                                                                            #  51    0xb39e0  2      OPC=movl_r32_r32    
  testq %rax, %rax                                                                           #  52    0xb39e2  3      OPC=testq_r64_r64   
  je .L_b3a20                                                                                #  53    0xb39e5  2      OPC=je_label        
  popq %rbx                                                                                  #  54    0xb39e7  1      OPC=popq_r64_1      
  popq %r11                                                                                  #  55    0xb39e8  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d                                                                    #  56    0xb39ea  7      OPC=andl_r32_imm32  
  nop                                                                                        #  57    0xb39f1  1      OPC=nop             
  nop                                                                                        #  58    0xb39f2  1      OPC=nop             
  nop                                                                                        #  59    0xb39f3  1      OPC=nop             
  nop                                                                                        #  60    0xb39f4  1      OPC=nop             
  addq %r15, %r11                                                                            #  61    0xb39f5  3      OPC=addq_r64_r64    
  jmpq %r11                                                                                  #  62    0xb39f8  3      OPC=jmpq_r64        
  nop                                                                                        #  63    0xb39fb  1      OPC=nop             
  nop                                                                                        #  64    0xb39fc  1      OPC=nop             
  nop                                                                                        #  65    0xb39fd  1      OPC=nop             
  nop                                                                                        #  66    0xb39fe  1      OPC=nop             
  nop                                                                                        #  67    0xb39ff  1      OPC=nop             
  nop                                                                                        #  68    0xb3a00  1      OPC=nop             
  nop                                                                                        #  69    0xb3a01  1      OPC=nop             
  nop                                                                                        #  70    0xb3a02  1      OPC=nop             
  nop                                                                                        #  71    0xb3a03  1      OPC=nop             
  nop                                                                                        #  72    0xb3a04  1      OPC=nop             
  nop                                                                                        #  73    0xb3a05  1      OPC=nop             
  nop                                                                                        #  74    0xb3a06  1      OPC=nop             
.L_b3a00:                                                                                    #        0xb3a07  0      OPC=<label>         
  nop                                                                                        #  75    0xb3a07  1      OPC=nop             
  nop                                                                                        #  76    0xb3a08  1      OPC=nop             
  nop                                                                                        #  77    0xb3a09  1      OPC=nop             
  nop                                                                                        #  78    0xb3a0a  1      OPC=nop             
  nop                                                                                        #  79    0xb3a0b  1      OPC=nop             
  nop                                                                                        #  80    0xb3a0c  1      OPC=nop             
  nop                                                                                        #  81    0xb3a0d  1      OPC=nop             
  nop                                                                                        #  82    0xb3a0e  1      OPC=nop             
  nop                                                                                        #  83    0xb3a0f  1      OPC=nop             
  nop                                                                                        #  84    0xb3a10  1      OPC=nop             
  nop                                                                                        #  85    0xb3a11  1      OPC=nop             
  nop                                                                                        #  86    0xb3a12  1      OPC=nop             
  nop                                                                                        #  87    0xb3a13  1      OPC=nop             
  nop                                                                                        #  88    0xb3a14  1      OPC=nop             
  nop                                                                                        #  89    0xb3a15  1      OPC=nop             
  nop                                                                                        #  90    0xb3a16  1      OPC=nop             
  nop                                                                                        #  91    0xb3a17  1      OPC=nop             
  nop                                                                                        #  92    0xb3a18  1      OPC=nop             
  nop                                                                                        #  93    0xb3a19  1      OPC=nop             
  nop                                                                                        #  94    0xb3a1a  1      OPC=nop             
  nop                                                                                        #  95    0xb3a1b  1      OPC=nop             
  nop                                                                                        #  96    0xb3a1c  1      OPC=nop             
  nop                                                                                        #  97    0xb3a1d  1      OPC=nop             
  nop                                                                                        #  98    0xb3a1e  1      OPC=nop             
  nop                                                                                        #  99    0xb3a1f  1      OPC=nop             
  nop                                                                                        #  100   0xb3a20  1      OPC=nop             
  nop                                                                                        #  101   0xb3a21  1      OPC=nop             
  callq ._ZSt16__throw_bad_castv                                                             #  102   0xb3a22  5      OPC=callq_label     
.L_b3a20:                                                                                    #        0xb3a27  0      OPC=<label>         
  nop                                                                                        #  103   0xb3a27  1      OPC=nop             
  nop                                                                                        #  104   0xb3a28  1      OPC=nop             
  nop                                                                                        #  105   0xb3a29  1      OPC=nop             
  nop                                                                                        #  106   0xb3a2a  1      OPC=nop             
  nop                                                                                        #  107   0xb3a2b  1      OPC=nop             
  nop                                                                                        #  108   0xb3a2c  1      OPC=nop             
  nop                                                                                        #  109   0xb3a2d  1      OPC=nop             
  nop                                                                                        #  110   0xb3a2e  1      OPC=nop             
  nop                                                                                        #  111   0xb3a2f  1      OPC=nop             
  nop                                                                                        #  112   0xb3a30  1      OPC=nop             
  nop                                                                                        #  113   0xb3a31  1      OPC=nop             
  nop                                                                                        #  114   0xb3a32  1      OPC=nop             
  nop                                                                                        #  115   0xb3a33  1      OPC=nop             
  nop                                                                                        #  116   0xb3a34  1      OPC=nop             
  nop                                                                                        #  117   0xb3a35  1      OPC=nop             
  nop                                                                                        #  118   0xb3a36  1      OPC=nop             
  nop                                                                                        #  119   0xb3a37  1      OPC=nop             
  nop                                                                                        #  120   0xb3a38  1      OPC=nop             
  nop                                                                                        #  121   0xb3a39  1      OPC=nop             
  nop                                                                                        #  122   0xb3a3a  1      OPC=nop             
  nop                                                                                        #  123   0xb3a3b  1      OPC=nop             
  nop                                                                                        #  124   0xb3a3c  1      OPC=nop             
  nop                                                                                        #  125   0xb3a3d  1      OPC=nop             
  nop                                                                                        #  126   0xb3a3e  1      OPC=nop             
  nop                                                                                        #  127   0xb3a3f  1      OPC=nop             
  nop                                                                                        #  128   0xb3a40  1      OPC=nop             
  nop                                                                                        #  129   0xb3a41  1      OPC=nop             
  callq .__cxa_bad_cast                                                                      #  130   0xb3a42  5      OPC=callq_label     
                                                                                                                                          
.size _ZSt9use_facetISt8time_getIwSt19istreambuf_iteratorIwSt11char_traitsIwEEEERKT_RKSt6locale, .-_ZSt9use_facetISt8time_getIwSt19istreambuf_iteratorIwSt11char_traitsIwEEEERKT_RKSt6locale

