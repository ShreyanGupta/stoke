  .text
  .globl _ZSt9use_facetISt8numpunctIwEERKT_RKSt6locale
  .type _ZSt9use_facetISt8numpunctIwEERKT_RKSt6locale, @function

#! file-offset 0xf4040
#! rip-offset  0xb4040
#! capacity    192 bytes

# Text                                           #  Line  RIP      Bytes  Opcode              
._ZSt9use_facetISt8numpunctIwEERKT_RKSt6locale:  #        0xb4040  0      OPC=<label>         
  pushq %rbx                                     #  1     0xb4040  1      OPC=pushq_r64_1     
  movl %edi, %ebx                                #  2     0xb4041  2      OPC=movl_r32_r32    
  movl $0x100735a0, %edi                         #  3     0xb4043  5      OPC=movl_r32_imm32  
  nop                                            #  4     0xb4048  1      OPC=nop             
  nop                                            #  5     0xb4049  1      OPC=nop             
  nop                                            #  6     0xb404a  1      OPC=nop             
  nop                                            #  7     0xb404b  1      OPC=nop             
  nop                                            #  8     0xb404c  1      OPC=nop             
  nop                                            #  9     0xb404d  1      OPC=nop             
  nop                                            #  10    0xb404e  1      OPC=nop             
  nop                                            #  11    0xb404f  1      OPC=nop             
  nop                                            #  12    0xb4050  1      OPC=nop             
  nop                                            #  13    0xb4051  1      OPC=nop             
  nop                                            #  14    0xb4052  1      OPC=nop             
  nop                                            #  15    0xb4053  1      OPC=nop             
  nop                                            #  16    0xb4054  1      OPC=nop             
  nop                                            #  17    0xb4055  1      OPC=nop             
  nop                                            #  18    0xb4056  1      OPC=nop             
  nop                                            #  19    0xb4057  1      OPC=nop             
  nop                                            #  20    0xb4058  1      OPC=nop             
  nop                                            #  21    0xb4059  1      OPC=nop             
  nop                                            #  22    0xb405a  1      OPC=nop             
  callq ._ZNKSt6locale2id5_M_idEv                #  23    0xb405b  5      OPC=callq_label     
  movl %ebx, %ebx                                #  24    0xb4060  2      OPC=movl_r32_r32    
  movl (%r15,%rbx,1), %edx                       #  25    0xb4062  4      OPC=movl_r32_m32    
  movl %edx, %edx                                #  26    0xb4066  2      OPC=movl_r32_r32    
  cmpl 0x8(%r15,%rdx,1), %eax                    #  27    0xb4068  5      OPC=cmpl_r32_m32    
  movl %edx, %edx                                #  28    0xb406d  2      OPC=movl_r32_r32    
  movl 0x4(%r15,%rdx,1), %ecx                    #  29    0xb406f  5      OPC=movl_r32_m32    
  jae .L_b40c0                                   #  30    0xb4074  2      OPC=jae_label       
  leal (%rcx,%rax,4), %eax                       #  31    0xb4076  3      OPC=leal_r32_m16    
  movl %eax, %eax                                #  32    0xb4079  2      OPC=movl_r32_r32    
  movl (%r15,%rax,1), %edi                       #  33    0xb407b  4      OPC=movl_r32_m32    
  nop                                            #  34    0xb407f  1      OPC=nop             
  testq %rdi, %rdi                               #  35    0xb4080  3      OPC=testq_r64_r64   
  je .L_b40c0                                    #  36    0xb4083  2      OPC=je_label        
  xorl %ecx, %ecx                                #  37    0xb4085  2      OPC=xorl_r32_r32    
  movl $0x1003c8e4, %edx                         #  38    0xb4087  5      OPC=movl_r32_imm32  
  movl $0x1003a2f4, %esi                         #  39    0xb408c  5      OPC=movl_r32_imm32  
  nop                                            #  40    0xb4091  1      OPC=nop             
  nop                                            #  41    0xb4092  1      OPC=nop             
  nop                                            #  42    0xb4093  1      OPC=nop             
  nop                                            #  43    0xb4094  1      OPC=nop             
  nop                                            #  44    0xb4095  1      OPC=nop             
  nop                                            #  45    0xb4096  1      OPC=nop             
  nop                                            #  46    0xb4097  1      OPC=nop             
  nop                                            #  47    0xb4098  1      OPC=nop             
  nop                                            #  48    0xb4099  1      OPC=nop             
  nop                                            #  49    0xb409a  1      OPC=nop             
  callq .__dynamic_cast                          #  50    0xb409b  5      OPC=callq_label     
  movl %eax, %eax                                #  51    0xb40a0  2      OPC=movl_r32_r32    
  testq %rax, %rax                               #  52    0xb40a2  3      OPC=testq_r64_r64   
  je .L_b40e0                                    #  53    0xb40a5  2      OPC=je_label        
  popq %rbx                                      #  54    0xb40a7  1      OPC=popq_r64_1      
  popq %r11                                      #  55    0xb40a8  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d                        #  56    0xb40aa  7      OPC=andl_r32_imm32  
  nop                                            #  57    0xb40b1  1      OPC=nop             
  nop                                            #  58    0xb40b2  1      OPC=nop             
  nop                                            #  59    0xb40b3  1      OPC=nop             
  nop                                            #  60    0xb40b4  1      OPC=nop             
  addq %r15, %r11                                #  61    0xb40b5  3      OPC=addq_r64_r64    
  jmpq %r11                                      #  62    0xb40b8  3      OPC=jmpq_r64        
  nop                                            #  63    0xb40bb  1      OPC=nop             
  nop                                            #  64    0xb40bc  1      OPC=nop             
  nop                                            #  65    0xb40bd  1      OPC=nop             
  nop                                            #  66    0xb40be  1      OPC=nop             
  nop                                            #  67    0xb40bf  1      OPC=nop             
  nop                                            #  68    0xb40c0  1      OPC=nop             
  nop                                            #  69    0xb40c1  1      OPC=nop             
  nop                                            #  70    0xb40c2  1      OPC=nop             
  nop                                            #  71    0xb40c3  1      OPC=nop             
  nop                                            #  72    0xb40c4  1      OPC=nop             
  nop                                            #  73    0xb40c5  1      OPC=nop             
  nop                                            #  74    0xb40c6  1      OPC=nop             
.L_b40c0:                                        #        0xb40c7  0      OPC=<label>         
  nop                                            #  75    0xb40c7  1      OPC=nop             
  nop                                            #  76    0xb40c8  1      OPC=nop             
  nop                                            #  77    0xb40c9  1      OPC=nop             
  nop                                            #  78    0xb40ca  1      OPC=nop             
  nop                                            #  79    0xb40cb  1      OPC=nop             
  nop                                            #  80    0xb40cc  1      OPC=nop             
  nop                                            #  81    0xb40cd  1      OPC=nop             
  nop                                            #  82    0xb40ce  1      OPC=nop             
  nop                                            #  83    0xb40cf  1      OPC=nop             
  nop                                            #  84    0xb40d0  1      OPC=nop             
  nop                                            #  85    0xb40d1  1      OPC=nop             
  nop                                            #  86    0xb40d2  1      OPC=nop             
  nop                                            #  87    0xb40d3  1      OPC=nop             
  nop                                            #  88    0xb40d4  1      OPC=nop             
  nop                                            #  89    0xb40d5  1      OPC=nop             
  nop                                            #  90    0xb40d6  1      OPC=nop             
  nop                                            #  91    0xb40d7  1      OPC=nop             
  nop                                            #  92    0xb40d8  1      OPC=nop             
  nop                                            #  93    0xb40d9  1      OPC=nop             
  nop                                            #  94    0xb40da  1      OPC=nop             
  nop                                            #  95    0xb40db  1      OPC=nop             
  nop                                            #  96    0xb40dc  1      OPC=nop             
  nop                                            #  97    0xb40dd  1      OPC=nop             
  nop                                            #  98    0xb40de  1      OPC=nop             
  nop                                            #  99    0xb40df  1      OPC=nop             
  nop                                            #  100   0xb40e0  1      OPC=nop             
  nop                                            #  101   0xb40e1  1      OPC=nop             
  callq ._ZSt16__throw_bad_castv                 #  102   0xb40e2  5      OPC=callq_label     
.L_b40e0:                                        #        0xb40e7  0      OPC=<label>         
  nop                                            #  103   0xb40e7  1      OPC=nop             
  nop                                            #  104   0xb40e8  1      OPC=nop             
  nop                                            #  105   0xb40e9  1      OPC=nop             
  nop                                            #  106   0xb40ea  1      OPC=nop             
  nop                                            #  107   0xb40eb  1      OPC=nop             
  nop                                            #  108   0xb40ec  1      OPC=nop             
  nop                                            #  109   0xb40ed  1      OPC=nop             
  nop                                            #  110   0xb40ee  1      OPC=nop             
  nop                                            #  111   0xb40ef  1      OPC=nop             
  nop                                            #  112   0xb40f0  1      OPC=nop             
  nop                                            #  113   0xb40f1  1      OPC=nop             
  nop                                            #  114   0xb40f2  1      OPC=nop             
  nop                                            #  115   0xb40f3  1      OPC=nop             
  nop                                            #  116   0xb40f4  1      OPC=nop             
  nop                                            #  117   0xb40f5  1      OPC=nop             
  nop                                            #  118   0xb40f6  1      OPC=nop             
  nop                                            #  119   0xb40f7  1      OPC=nop             
  nop                                            #  120   0xb40f8  1      OPC=nop             
  nop                                            #  121   0xb40f9  1      OPC=nop             
  nop                                            #  122   0xb40fa  1      OPC=nop             
  nop                                            #  123   0xb40fb  1      OPC=nop             
  nop                                            #  124   0xb40fc  1      OPC=nop             
  nop                                            #  125   0xb40fd  1      OPC=nop             
  nop                                            #  126   0xb40fe  1      OPC=nop             
  nop                                            #  127   0xb40ff  1      OPC=nop             
  nop                                            #  128   0xb4100  1      OPC=nop             
  nop                                            #  129   0xb4101  1      OPC=nop             
  callq .__cxa_bad_cast                          #  130   0xb4102  5      OPC=callq_label     
                                                                                              
.size _ZSt9use_facetISt8numpunctIwEERKT_RKSt6locale, .-_ZSt9use_facetISt8numpunctIwEERKT_RKSt6locale

