  .text
  .globl _ZNSbIwSt11char_traitsIwESaIwEE4_Rep8_M_cloneERKS1_j
  .type _ZNSbIwSt11char_traitsIwESaIwEE4_Rep8_M_cloneERKS1_j, @function

#! file-offset 0x116ca0
#! rip-offset  0xd6ca0
#! capacity    288 bytes

# Text                                                           #  Line  RIP      Bytes  Opcode              
._ZNSbIwSt11char_traitsIwESaIwEE4_Rep8_M_cloneERKS1_j:           #        0xd6ca0  0      OPC=<label>         
  movq %rbx, -0x18(%rsp)                                         #  1     0xd6ca0  5      OPC=movq_m64_r64    
  movl %edi, %ebx                                                #  2     0xd6ca5  2      OPC=movl_r32_r32    
  movq %r12, -0x10(%rsp)                                         #  3     0xd6ca7  5      OPC=movq_m64_r64    
  movl %ebx, %ebx                                                #  4     0xd6cac  2      OPC=movl_r32_r32    
  movq %r13, -0x8(%rsp)                                          #  5     0xd6cae  5      OPC=movq_m64_r64    
  movl %edx, %edi                                                #  6     0xd6cb3  2      OPC=movl_r32_r32    
  subl $0x18, %esp                                               #  7     0xd6cb5  3      OPC=subl_r32_imm8   
  addq %r15, %rsp                                                #  8     0xd6cb8  3      OPC=addq_r64_r64    
  nop                                                            #  9     0xd6cbb  1      OPC=nop             
  nop                                                            #  10    0xd6cbc  1      OPC=nop             
  nop                                                            #  11    0xd6cbd  1      OPC=nop             
  nop                                                            #  12    0xd6cbe  1      OPC=nop             
  nop                                                            #  13    0xd6cbf  1      OPC=nop             
  movl %ebx, %ebx                                                #  14    0xd6cc0  2      OPC=movl_r32_r32    
  addl (%r15,%rbx,1), %edi                                       #  15    0xd6cc2  4      OPC=addl_r32_m32    
  movl %esi, %edx                                                #  16    0xd6cc6  2      OPC=movl_r32_r32    
  movl %ebx, %ebx                                                #  17    0xd6cc8  2      OPC=movl_r32_r32    
  movl 0x4(%r15,%rbx,1), %esi                                    #  18    0xd6cca  5      OPC=movl_r32_m32    
  nop                                                            #  19    0xd6ccf  1      OPC=nop             
  nop                                                            #  20    0xd6cd0  1      OPC=nop             
  nop                                                            #  21    0xd6cd1  1      OPC=nop             
  nop                                                            #  22    0xd6cd2  1      OPC=nop             
  nop                                                            #  23    0xd6cd3  1      OPC=nop             
  nop                                                            #  24    0xd6cd4  1      OPC=nop             
  nop                                                            #  25    0xd6cd5  1      OPC=nop             
  nop                                                            #  26    0xd6cd6  1      OPC=nop             
  nop                                                            #  27    0xd6cd7  1      OPC=nop             
  nop                                                            #  28    0xd6cd8  1      OPC=nop             
  nop                                                            #  29    0xd6cd9  1      OPC=nop             
  nop                                                            #  30    0xd6cda  1      OPC=nop             
  callq ._ZNSbIwSt11char_traitsIwESaIwEE4_Rep9_S_createEjjRKS1_  #  31    0xd6cdb  5      OPC=callq_label     
  movl %ebx, %ebx                                                #  32    0xd6ce0  2      OPC=movl_r32_r32    
  movl (%r15,%rbx,1), %edx                                       #  33    0xd6ce2  4      OPC=movl_r32_m32    
  movl %eax, %r12d                                               #  34    0xd6ce6  3      OPC=movl_r32_r32    
  testl %edx, %edx                                               #  35    0xd6ce9  2      OPC=testl_r32_r32   
  jne .L_d6d60                                                   #  36    0xd6ceb  2      OPC=jne_label       
  leal 0xc(%r12), %r13d                                          #  37    0xd6ced  5      OPC=leal_r32_m16    
  xorl %eax, %eax                                                #  38    0xd6cf2  2      OPC=xorl_r32_r32    
  nop                                                            #  39    0xd6cf4  1      OPC=nop             
  nop                                                            #  40    0xd6cf5  1      OPC=nop             
  nop                                                            #  41    0xd6cf6  1      OPC=nop             
  nop                                                            #  42    0xd6cf7  1      OPC=nop             
  nop                                                            #  43    0xd6cf8  1      OPC=nop             
  nop                                                            #  44    0xd6cf9  1      OPC=nop             
  nop                                                            #  45    0xd6cfa  1      OPC=nop             
  nop                                                            #  46    0xd6cfb  1      OPC=nop             
  nop                                                            #  47    0xd6cfc  1      OPC=nop             
  nop                                                            #  48    0xd6cfd  1      OPC=nop             
  nop                                                            #  49    0xd6cfe  1      OPC=nop             
  nop                                                            #  50    0xd6cff  1      OPC=nop             
.L_d6d00:                                                        #        0xd6d00  0      OPC=<label>         
  movl %r12d, %r12d                                              #  51    0xd6d00  3      OPC=movl_r32_r32    
  movl %eax, (%r15,%r12,1)                                       #  52    0xd6d03  4      OPC=movl_m32_r32    
  leal (%r13,%rax,4), %eax                                       #  53    0xd6d07  5      OPC=leal_r32_m16    
  movl %r12d, %r12d                                              #  54    0xd6d0c  3      OPC=movl_r32_r32    
  movl $0x0, 0x8(%r15,%r12,1)                                    #  55    0xd6d0f  9      OPC=movl_m32_imm32  
  nop                                                            #  56    0xd6d18  1      OPC=nop             
  nop                                                            #  57    0xd6d19  1      OPC=nop             
  nop                                                            #  58    0xd6d1a  1      OPC=nop             
  nop                                                            #  59    0xd6d1b  1      OPC=nop             
  nop                                                            #  60    0xd6d1c  1      OPC=nop             
  nop                                                            #  61    0xd6d1d  1      OPC=nop             
  nop                                                            #  62    0xd6d1e  1      OPC=nop             
  nop                                                            #  63    0xd6d1f  1      OPC=nop             
  movl %eax, %eax                                                #  64    0xd6d20  2      OPC=movl_r32_r32    
  movl $0x0, (%r15,%rax,1)                                       #  65    0xd6d22  8      OPC=movl_m32_imm32  
  movl %r13d, %eax                                               #  66    0xd6d2a  3      OPC=movl_r32_r32    
  movq (%rsp), %rbx                                              #  67    0xd6d2d  4      OPC=movq_r64_m64    
  movq 0x8(%rsp), %r12                                           #  68    0xd6d31  5      OPC=movq_r64_m64    
  movq 0x10(%rsp), %r13                                          #  69    0xd6d36  5      OPC=movq_r64_m64    
  nop                                                            #  70    0xd6d3b  1      OPC=nop             
  nop                                                            #  71    0xd6d3c  1      OPC=nop             
  nop                                                            #  72    0xd6d3d  1      OPC=nop             
  nop                                                            #  73    0xd6d3e  1      OPC=nop             
  nop                                                            #  74    0xd6d3f  1      OPC=nop             
  addl $0x18, %esp                                               #  75    0xd6d40  3      OPC=addl_r32_imm8   
  addq %r15, %rsp                                                #  76    0xd6d43  3      OPC=addq_r64_r64    
  popq %r11                                                      #  77    0xd6d46  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d                                        #  78    0xd6d48  7      OPC=andl_r32_imm32  
  nop                                                            #  79    0xd6d4f  1      OPC=nop             
  nop                                                            #  80    0xd6d50  1      OPC=nop             
  nop                                                            #  81    0xd6d51  1      OPC=nop             
  nop                                                            #  82    0xd6d52  1      OPC=nop             
  addq %r15, %r11                                                #  83    0xd6d53  3      OPC=addq_r64_r64    
  jmpq %r11                                                      #  84    0xd6d56  3      OPC=jmpq_r64        
  nop                                                            #  85    0xd6d59  1      OPC=nop             
  nop                                                            #  86    0xd6d5a  1      OPC=nop             
  nop                                                            #  87    0xd6d5b  1      OPC=nop             
  nop                                                            #  88    0xd6d5c  1      OPC=nop             
  nop                                                            #  89    0xd6d5d  1      OPC=nop             
  nop                                                            #  90    0xd6d5e  1      OPC=nop             
  nop                                                            #  91    0xd6d5f  1      OPC=nop             
  nop                                                            #  92    0xd6d60  1      OPC=nop             
  nop                                                            #  93    0xd6d61  1      OPC=nop             
  nop                                                            #  94    0xd6d62  1      OPC=nop             
  nop                                                            #  95    0xd6d63  1      OPC=nop             
  nop                                                            #  96    0xd6d64  1      OPC=nop             
  nop                                                            #  97    0xd6d65  1      OPC=nop             
  nop                                                            #  98    0xd6d66  1      OPC=nop             
.L_d6d60:                                                        #        0xd6d67  0      OPC=<label>         
  cmpl $0x1, %edx                                                #  99    0xd6d67  3      OPC=cmpl_r32_imm8   
  leal 0xc(%rbx), %esi                                           #  100   0xd6d6a  3      OPC=leal_r32_m16    
  leal 0xc(%r12), %r13d                                          #  101   0xd6d6d  5      OPC=leal_r32_m16    
  je .L_d6da0                                                    #  102   0xd6d72  2      OPC=je_label        
  movl %r13d, %edi                                               #  103   0xd6d74  3      OPC=movl_r32_r32    
  nop                                                            #  104   0xd6d77  1      OPC=nop             
  nop                                                            #  105   0xd6d78  1      OPC=nop             
  nop                                                            #  106   0xd6d79  1      OPC=nop             
  nop                                                            #  107   0xd6d7a  1      OPC=nop             
  nop                                                            #  108   0xd6d7b  1      OPC=nop             
  nop                                                            #  109   0xd6d7c  1      OPC=nop             
  nop                                                            #  110   0xd6d7d  1      OPC=nop             
  nop                                                            #  111   0xd6d7e  1      OPC=nop             
  nop                                                            #  112   0xd6d7f  1      OPC=nop             
  nop                                                            #  113   0xd6d80  1      OPC=nop             
  nop                                                            #  114   0xd6d81  1      OPC=nop             
  callq .wmemcpy                                                 #  115   0xd6d82  5      OPC=callq_label     
  movl %ebx, %ebx                                                #  116   0xd6d87  2      OPC=movl_r32_r32    
  movl (%r15,%rbx,1), %eax                                       #  117   0xd6d89  4      OPC=movl_r32_m32    
  jmpq .L_d6d00                                                  #  118   0xd6d8d  5      OPC=jmpq_label_1    
  nop                                                            #  119   0xd6d92  1      OPC=nop             
  nop                                                            #  120   0xd6d93  1      OPC=nop             
  nop                                                            #  121   0xd6d94  1      OPC=nop             
  nop                                                            #  122   0xd6d95  1      OPC=nop             
  nop                                                            #  123   0xd6d96  1      OPC=nop             
  nop                                                            #  124   0xd6d97  1      OPC=nop             
  nop                                                            #  125   0xd6d98  1      OPC=nop             
  nop                                                            #  126   0xd6d99  1      OPC=nop             
  nop                                                            #  127   0xd6d9a  1      OPC=nop             
  nop                                                            #  128   0xd6d9b  1      OPC=nop             
  nop                                                            #  129   0xd6d9c  1      OPC=nop             
  nop                                                            #  130   0xd6d9d  1      OPC=nop             
  nop                                                            #  131   0xd6d9e  1      OPC=nop             
  nop                                                            #  132   0xd6d9f  1      OPC=nop             
  nop                                                            #  133   0xd6da0  1      OPC=nop             
  nop                                                            #  134   0xd6da1  1      OPC=nop             
  nop                                                            #  135   0xd6da2  1      OPC=nop             
  nop                                                            #  136   0xd6da3  1      OPC=nop             
  nop                                                            #  137   0xd6da4  1      OPC=nop             
  nop                                                            #  138   0xd6da5  1      OPC=nop             
  nop                                                            #  139   0xd6da6  1      OPC=nop             
.L_d6da0:                                                        #        0xd6da7  0      OPC=<label>         
  movl %esi, %esi                                                #  140   0xd6da7  2      OPC=movl_r32_r32    
  movl (%r15,%rsi,1), %eax                                       #  141   0xd6da9  4      OPC=movl_r32_m32    
  movl %r13d, %r13d                                              #  142   0xd6dad  3      OPC=movl_r32_r32    
  movl %eax, (%r15,%r13,1)                                       #  143   0xd6db0  4      OPC=movl_m32_r32    
  movl $0x1, %eax                                                #  144   0xd6db4  5      OPC=movl_r32_imm32  
  jmpq .L_d6d00                                                  #  145   0xd6db9  5      OPC=jmpq_label_1    
  nop                                                            #  146   0xd6dbe  1      OPC=nop             
  nop                                                            #  147   0xd6dbf  1      OPC=nop             
  nop                                                            #  148   0xd6dc0  1      OPC=nop             
  nop                                                            #  149   0xd6dc1  1      OPC=nop             
  nop                                                            #  150   0xd6dc2  1      OPC=nop             
  nop                                                            #  151   0xd6dc3  1      OPC=nop             
  nop                                                            #  152   0xd6dc4  1      OPC=nop             
  nop                                                            #  153   0xd6dc5  1      OPC=nop             
  nop                                                            #  154   0xd6dc6  1      OPC=nop             
                                                                                                              
.size _ZNSbIwSt11char_traitsIwESaIwEE4_Rep8_M_cloneERKS1_j, .-_ZNSbIwSt11char_traitsIwESaIwEE4_Rep8_M_cloneERKS1_j

