  .text
  .globl _ZSt9use_facetISt7codecvtIwc10_mbstate_tEERKT_RKSt6locale
  .type _ZSt9use_facetISt7codecvtIwc10_mbstate_tEERKT_RKSt6locale, @function

#! file-offset 0xf41c0
#! rip-offset  0xb41c0
#! capacity    192 bytes

# Text                                                       #  Line  RIP      Bytes  Opcode              
._ZSt9use_facetISt7codecvtIwc10_mbstate_tEERKT_RKSt6locale:  #        0xb41c0  0      OPC=<label>         
  pushq %rbx                                                 #  1     0xb41c0  1      OPC=pushq_r64_1     
  movl %edi, %ebx                                            #  2     0xb41c1  2      OPC=movl_r32_r32    
  movl $0x100780a8, %edi                                     #  3     0xb41c3  5      OPC=movl_r32_imm32  
  nop                                                        #  4     0xb41c8  1      OPC=nop             
  nop                                                        #  5     0xb41c9  1      OPC=nop             
  nop                                                        #  6     0xb41ca  1      OPC=nop             
  nop                                                        #  7     0xb41cb  1      OPC=nop             
  nop                                                        #  8     0xb41cc  1      OPC=nop             
  nop                                                        #  9     0xb41cd  1      OPC=nop             
  nop                                                        #  10    0xb41ce  1      OPC=nop             
  nop                                                        #  11    0xb41cf  1      OPC=nop             
  nop                                                        #  12    0xb41d0  1      OPC=nop             
  nop                                                        #  13    0xb41d1  1      OPC=nop             
  nop                                                        #  14    0xb41d2  1      OPC=nop             
  nop                                                        #  15    0xb41d3  1      OPC=nop             
  nop                                                        #  16    0xb41d4  1      OPC=nop             
  nop                                                        #  17    0xb41d5  1      OPC=nop             
  nop                                                        #  18    0xb41d6  1      OPC=nop             
  nop                                                        #  19    0xb41d7  1      OPC=nop             
  nop                                                        #  20    0xb41d8  1      OPC=nop             
  nop                                                        #  21    0xb41d9  1      OPC=nop             
  nop                                                        #  22    0xb41da  1      OPC=nop             
  callq ._ZNKSt6locale2id5_M_idEv                            #  23    0xb41db  5      OPC=callq_label     
  movl %ebx, %ebx                                            #  24    0xb41e0  2      OPC=movl_r32_r32    
  movl (%r15,%rbx,1), %edx                                   #  25    0xb41e2  4      OPC=movl_r32_m32    
  movl %edx, %edx                                            #  26    0xb41e6  2      OPC=movl_r32_r32    
  cmpl 0x8(%r15,%rdx,1), %eax                                #  27    0xb41e8  5      OPC=cmpl_r32_m32    
  movl %edx, %edx                                            #  28    0xb41ed  2      OPC=movl_r32_r32    
  movl 0x4(%r15,%rdx,1), %ecx                                #  29    0xb41ef  5      OPC=movl_r32_m32    
  jae .L_b4240                                               #  30    0xb41f4  2      OPC=jae_label       
  leal (%rcx,%rax,4), %eax                                   #  31    0xb41f6  3      OPC=leal_r32_m16    
  movl %eax, %eax                                            #  32    0xb41f9  2      OPC=movl_r32_r32    
  movl (%r15,%rax,1), %edi                                   #  33    0xb41fb  4      OPC=movl_r32_m32    
  nop                                                        #  34    0xb41ff  1      OPC=nop             
  testq %rdi, %rdi                                           #  35    0xb4200  3      OPC=testq_r64_r64   
  je .L_b4240                                                #  36    0xb4203  2      OPC=je_label        
  xorl %ecx, %ecx                                            #  37    0xb4205  2      OPC=xorl_r32_r32    
  movl $0x1003d52c, %edx                                     #  38    0xb4207  5      OPC=movl_r32_imm32  
  movl $0x1003a2f4, %esi                                     #  39    0xb420c  5      OPC=movl_r32_imm32  
  nop                                                        #  40    0xb4211  1      OPC=nop             
  nop                                                        #  41    0xb4212  1      OPC=nop             
  nop                                                        #  42    0xb4213  1      OPC=nop             
  nop                                                        #  43    0xb4214  1      OPC=nop             
  nop                                                        #  44    0xb4215  1      OPC=nop             
  nop                                                        #  45    0xb4216  1      OPC=nop             
  nop                                                        #  46    0xb4217  1      OPC=nop             
  nop                                                        #  47    0xb4218  1      OPC=nop             
  nop                                                        #  48    0xb4219  1      OPC=nop             
  nop                                                        #  49    0xb421a  1      OPC=nop             
  callq .__dynamic_cast                                      #  50    0xb421b  5      OPC=callq_label     
  movl %eax, %eax                                            #  51    0xb4220  2      OPC=movl_r32_r32    
  testq %rax, %rax                                           #  52    0xb4222  3      OPC=testq_r64_r64   
  je .L_b4260                                                #  53    0xb4225  2      OPC=je_label        
  popq %rbx                                                  #  54    0xb4227  1      OPC=popq_r64_1      
  popq %r11                                                  #  55    0xb4228  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d                                    #  56    0xb422a  7      OPC=andl_r32_imm32  
  nop                                                        #  57    0xb4231  1      OPC=nop             
  nop                                                        #  58    0xb4232  1      OPC=nop             
  nop                                                        #  59    0xb4233  1      OPC=nop             
  nop                                                        #  60    0xb4234  1      OPC=nop             
  addq %r15, %r11                                            #  61    0xb4235  3      OPC=addq_r64_r64    
  jmpq %r11                                                  #  62    0xb4238  3      OPC=jmpq_r64        
  nop                                                        #  63    0xb423b  1      OPC=nop             
  nop                                                        #  64    0xb423c  1      OPC=nop             
  nop                                                        #  65    0xb423d  1      OPC=nop             
  nop                                                        #  66    0xb423e  1      OPC=nop             
  nop                                                        #  67    0xb423f  1      OPC=nop             
  nop                                                        #  68    0xb4240  1      OPC=nop             
  nop                                                        #  69    0xb4241  1      OPC=nop             
  nop                                                        #  70    0xb4242  1      OPC=nop             
  nop                                                        #  71    0xb4243  1      OPC=nop             
  nop                                                        #  72    0xb4244  1      OPC=nop             
  nop                                                        #  73    0xb4245  1      OPC=nop             
  nop                                                        #  74    0xb4246  1      OPC=nop             
.L_b4240:                                                    #        0xb4247  0      OPC=<label>         
  nop                                                        #  75    0xb4247  1      OPC=nop             
  nop                                                        #  76    0xb4248  1      OPC=nop             
  nop                                                        #  77    0xb4249  1      OPC=nop             
  nop                                                        #  78    0xb424a  1      OPC=nop             
  nop                                                        #  79    0xb424b  1      OPC=nop             
  nop                                                        #  80    0xb424c  1      OPC=nop             
  nop                                                        #  81    0xb424d  1      OPC=nop             
  nop                                                        #  82    0xb424e  1      OPC=nop             
  nop                                                        #  83    0xb424f  1      OPC=nop             
  nop                                                        #  84    0xb4250  1      OPC=nop             
  nop                                                        #  85    0xb4251  1      OPC=nop             
  nop                                                        #  86    0xb4252  1      OPC=nop             
  nop                                                        #  87    0xb4253  1      OPC=nop             
  nop                                                        #  88    0xb4254  1      OPC=nop             
  nop                                                        #  89    0xb4255  1      OPC=nop             
  nop                                                        #  90    0xb4256  1      OPC=nop             
  nop                                                        #  91    0xb4257  1      OPC=nop             
  nop                                                        #  92    0xb4258  1      OPC=nop             
  nop                                                        #  93    0xb4259  1      OPC=nop             
  nop                                                        #  94    0xb425a  1      OPC=nop             
  nop                                                        #  95    0xb425b  1      OPC=nop             
  nop                                                        #  96    0xb425c  1      OPC=nop             
  nop                                                        #  97    0xb425d  1      OPC=nop             
  nop                                                        #  98    0xb425e  1      OPC=nop             
  nop                                                        #  99    0xb425f  1      OPC=nop             
  nop                                                        #  100   0xb4260  1      OPC=nop             
  nop                                                        #  101   0xb4261  1      OPC=nop             
  callq ._ZSt16__throw_bad_castv                             #  102   0xb4262  5      OPC=callq_label     
.L_b4260:                                                    #        0xb4267  0      OPC=<label>         
  nop                                                        #  103   0xb4267  1      OPC=nop             
  nop                                                        #  104   0xb4268  1      OPC=nop             
  nop                                                        #  105   0xb4269  1      OPC=nop             
  nop                                                        #  106   0xb426a  1      OPC=nop             
  nop                                                        #  107   0xb426b  1      OPC=nop             
  nop                                                        #  108   0xb426c  1      OPC=nop             
  nop                                                        #  109   0xb426d  1      OPC=nop             
  nop                                                        #  110   0xb426e  1      OPC=nop             
  nop                                                        #  111   0xb426f  1      OPC=nop             
  nop                                                        #  112   0xb4270  1      OPC=nop             
  nop                                                        #  113   0xb4271  1      OPC=nop             
  nop                                                        #  114   0xb4272  1      OPC=nop             
  nop                                                        #  115   0xb4273  1      OPC=nop             
  nop                                                        #  116   0xb4274  1      OPC=nop             
  nop                                                        #  117   0xb4275  1      OPC=nop             
  nop                                                        #  118   0xb4276  1      OPC=nop             
  nop                                                        #  119   0xb4277  1      OPC=nop             
  nop                                                        #  120   0xb4278  1      OPC=nop             
  nop                                                        #  121   0xb4279  1      OPC=nop             
  nop                                                        #  122   0xb427a  1      OPC=nop             
  nop                                                        #  123   0xb427b  1      OPC=nop             
  nop                                                        #  124   0xb427c  1      OPC=nop             
  nop                                                        #  125   0xb427d  1      OPC=nop             
  nop                                                        #  126   0xb427e  1      OPC=nop             
  nop                                                        #  127   0xb427f  1      OPC=nop             
  nop                                                        #  128   0xb4280  1      OPC=nop             
  nop                                                        #  129   0xb4281  1      OPC=nop             
  callq .__cxa_bad_cast                                      #  130   0xb4282  5      OPC=callq_label     
                                                                                                          
.size _ZSt9use_facetISt7codecvtIwc10_mbstate_tEERKT_RKSt6locale, .-_ZSt9use_facetISt7codecvtIwc10_mbstate_tEERKT_RKSt6locale

