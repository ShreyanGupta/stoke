  .text
  .globl _ZNKSbIwSt11char_traitsIwESaIwEE4findEPKwjj
  .type _ZNKSbIwSt11char_traitsIwESaIwEE4findEPKwjj, @function

#! file-offset 0x115e60
#! rip-offset  0xd5e60
#! capacity    320 bytes

# Text                                         #  Line  RIP      Bytes  Opcode                
._ZNKSbIwSt11char_traitsIwESaIwEE4findEPKwjj:  #        0xd5e60  0      OPC=<label>           
  pushq %r14                                   #  1     0xd5e60  2      OPC=pushq_r64_1       
  movl %edi, %edi                              #  2     0xd5e62  2      OPC=movl_r32_r32      
  movl %esi, %r14d                             #  3     0xd5e64  3      OPC=movl_r32_r32      
  pushq %r13                                   #  4     0xd5e67  2      OPC=pushq_r64_1       
  pushq %r12                                   #  5     0xd5e69  2      OPC=pushq_r64_1       
  pushq %rbx                                   #  6     0xd5e6b  1      OPC=pushq_r64_1       
  movl %edx, %ebx                              #  7     0xd5e6c  2      OPC=movl_r32_r32      
  subl $0x18, %esp                             #  8     0xd5e6e  3      OPC=subl_r32_imm8     
  addq %r15, %rsp                              #  9     0xd5e71  3      OPC=addq_r64_r64      
  movl %edi, %edi                              #  10    0xd5e74  2      OPC=movl_r32_r32      
  movl (%r15,%rdi,1), %r13d                    #  11    0xd5e76  4      OPC=movl_r32_m32      
  testl %ecx, %ecx                             #  12    0xd5e7a  2      OPC=testl_r32_r32     
  leal -0xc(%r13), %eax                        #  13    0xd5e7c  4      OPC=leal_r32_m16      
  movl %eax, %eax                              #  14    0xd5e80  2      OPC=movl_r32_r32      
  movl (%r15,%rax,1), %r8d                     #  15    0xd5e82  4      OPC=movl_r32_m32      
  je .L_d5f60                                  #  16    0xd5e86  6      OPC=je_label_1        
  cmpl %r8d, %ecx                              #  17    0xd5e8c  3      OPC=cmpl_r32_r32      
  ja .L_d5f80                                  #  18    0xd5e8f  6      OPC=ja_label_1        
  subl %ecx, %r8d                              #  19    0xd5e95  3      OPC=subl_r32_r32      
  cmpl %edx, %r8d                              #  20    0xd5e98  3      OPC=cmpl_r32_r32      
  nop                                          #  21    0xd5e9b  1      OPC=nop               
  nop                                          #  22    0xd5e9c  1      OPC=nop               
  nop                                          #  23    0xd5e9d  1      OPC=nop               
  nop                                          #  24    0xd5e9e  1      OPC=nop               
  nop                                          #  25    0xd5e9f  1      OPC=nop               
  jb .L_d5f80                                  #  26    0xd5ea0  6      OPC=jb_label_1        
  leal 0x4(%r14), %eax                         #  27    0xd5ea6  4      OPC=leal_r32_m16      
  subl $0x1, %ecx                              #  28    0xd5eaa  3      OPC=subl_r32_imm8     
  movl %ecx, 0x8(%rsp)                         #  29    0xd5ead  4      OPC=movl_m32_r32      
  movl %eax, 0xc(%rsp)                         #  30    0xd5eb1  4      OPC=movl_m32_r32      
  jmpq .L_d5ee0                                #  31    0xd5eb5  2      OPC=jmpq_label        
  nop                                          #  32    0xd5eb7  1      OPC=nop               
  nop                                          #  33    0xd5eb8  1      OPC=nop               
  nop                                          #  34    0xd5eb9  1      OPC=nop               
  nop                                          #  35    0xd5eba  1      OPC=nop               
  nop                                          #  36    0xd5ebb  1      OPC=nop               
  nop                                          #  37    0xd5ebc  1      OPC=nop               
  nop                                          #  38    0xd5ebd  1      OPC=nop               
  nop                                          #  39    0xd5ebe  1      OPC=nop               
  nop                                          #  40    0xd5ebf  1      OPC=nop               
.L_d5ec0:                                      #        0xd5ec0  0      OPC=<label>           
  cmpl %r12d, %r8d                             #  41    0xd5ec0  3      OPC=cmpl_r32_r32      
  jb .L_d5f80                                  #  42    0xd5ec3  6      OPC=jb_label_1        
  movl %r12d, %ebx                             #  43    0xd5ec9  3      OPC=movl_r32_r32      
  nop                                          #  44    0xd5ecc  1      OPC=nop               
  nop                                          #  45    0xd5ecd  1      OPC=nop               
  nop                                          #  46    0xd5ece  1      OPC=nop               
  nop                                          #  47    0xd5ecf  1      OPC=nop               
  nop                                          #  48    0xd5ed0  1      OPC=nop               
  nop                                          #  49    0xd5ed1  1      OPC=nop               
  nop                                          #  50    0xd5ed2  1      OPC=nop               
  nop                                          #  51    0xd5ed3  1      OPC=nop               
  nop                                          #  52    0xd5ed4  1      OPC=nop               
  nop                                          #  53    0xd5ed5  1      OPC=nop               
  nop                                          #  54    0xd5ed6  1      OPC=nop               
  nop                                          #  55    0xd5ed7  1      OPC=nop               
  nop                                          #  56    0xd5ed8  1      OPC=nop               
  nop                                          #  57    0xd5ed9  1      OPC=nop               
  nop                                          #  58    0xd5eda  1      OPC=nop               
  nop                                          #  59    0xd5edb  1      OPC=nop               
  nop                                          #  60    0xd5edc  1      OPC=nop               
  nop                                          #  61    0xd5edd  1      OPC=nop               
  nop                                          #  62    0xd5ede  1      OPC=nop               
  nop                                          #  63    0xd5edf  1      OPC=nop               
.L_d5ee0:                                      #        0xd5ee0  0      OPC=<label>           
  leal (%r13,%rbx,4), %eax                     #  64    0xd5ee0  5      OPC=leal_r32_m16      
  leal 0x1(%rbx), %r12d                        #  65    0xd5ee5  4      OPC=leal_r32_m16      
  movl %eax, %eax                              #  66    0xd5ee9  2      OPC=movl_r32_r32      
  movl (%r15,%rax,1), %eax                     #  67    0xd5eeb  4      OPC=movl_r32_m32      
  movl %r14d, %r14d                            #  68    0xd5eef  3      OPC=movl_r32_r32      
  cmpl (%r15,%r14,1), %eax                     #  69    0xd5ef2  4      OPC=cmpl_r32_m32      
  jne .L_d5ec0                                 #  70    0xd5ef6  2      OPC=jne_label         
  leal 0x1(%rbx), %r12d                        #  71    0xd5ef8  4      OPC=leal_r32_m16      
  movl 0x8(%rsp), %edx                         #  72    0xd5efc  4      OPC=movl_r32_m32      
  movl 0xc(%rsp), %esi                         #  73    0xd5f00  4      OPC=movl_r32_m32      
  movl %r8d, (%rsp)                            #  74    0xd5f04  4      OPC=movl_m32_r32      
  leal (%r13,%r12,4), %edi                     #  75    0xd5f08  5      OPC=leal_r32_m16      
  nop                                          #  76    0xd5f0d  1      OPC=nop               
  nop                                          #  77    0xd5f0e  1      OPC=nop               
  nop                                          #  78    0xd5f0f  1      OPC=nop               
  nop                                          #  79    0xd5f10  1      OPC=nop               
  nop                                          #  80    0xd5f11  1      OPC=nop               
  nop                                          #  81    0xd5f12  1      OPC=nop               
  nop                                          #  82    0xd5f13  1      OPC=nop               
  nop                                          #  83    0xd5f14  1      OPC=nop               
  nop                                          #  84    0xd5f15  1      OPC=nop               
  nop                                          #  85    0xd5f16  1      OPC=nop               
  nop                                          #  86    0xd5f17  1      OPC=nop               
  nop                                          #  87    0xd5f18  1      OPC=nop               
  nop                                          #  88    0xd5f19  1      OPC=nop               
  nop                                          #  89    0xd5f1a  1      OPC=nop               
  callq .wmemcmp                               #  90    0xd5f1b  5      OPC=callq_label       
  testl %eax, %eax                             #  91    0xd5f20  2      OPC=testl_r32_r32     
  movl (%rsp), %r8d                            #  92    0xd5f22  4      OPC=movl_r32_m32      
  jne .L_d5ec0                                 #  93    0xd5f26  2      OPC=jne_label         
  nop                                          #  94    0xd5f28  1      OPC=nop               
  nop                                          #  95    0xd5f29  1      OPC=nop               
  nop                                          #  96    0xd5f2a  1      OPC=nop               
  nop                                          #  97    0xd5f2b  1      OPC=nop               
  nop                                          #  98    0xd5f2c  1      OPC=nop               
  nop                                          #  99    0xd5f2d  1      OPC=nop               
  nop                                          #  100   0xd5f2e  1      OPC=nop               
  nop                                          #  101   0xd5f2f  1      OPC=nop               
  nop                                          #  102   0xd5f30  1      OPC=nop               
  nop                                          #  103   0xd5f31  1      OPC=nop               
  nop                                          #  104   0xd5f32  1      OPC=nop               
  nop                                          #  105   0xd5f33  1      OPC=nop               
  nop                                          #  106   0xd5f34  1      OPC=nop               
  nop                                          #  107   0xd5f35  1      OPC=nop               
  nop                                          #  108   0xd5f36  1      OPC=nop               
  nop                                          #  109   0xd5f37  1      OPC=nop               
  nop                                          #  110   0xd5f38  1      OPC=nop               
  nop                                          #  111   0xd5f39  1      OPC=nop               
  nop                                          #  112   0xd5f3a  1      OPC=nop               
  nop                                          #  113   0xd5f3b  1      OPC=nop               
  nop                                          #  114   0xd5f3c  1      OPC=nop               
  nop                                          #  115   0xd5f3d  1      OPC=nop               
  nop                                          #  116   0xd5f3e  1      OPC=nop               
  nop                                          #  117   0xd5f3f  1      OPC=nop               
.L_d5f40:                                      #        0xd5f40  0      OPC=<label>           
  addl $0x18, %esp                             #  118   0xd5f40  3      OPC=addl_r32_imm8     
  addq %r15, %rsp                              #  119   0xd5f43  3      OPC=addq_r64_r64      
  movl %ebx, %eax                              #  120   0xd5f46  2      OPC=movl_r32_r32      
  popq %rbx                                    #  121   0xd5f48  1      OPC=popq_r64_1        
  popq %r12                                    #  122   0xd5f49  2      OPC=popq_r64_1        
  popq %r13                                    #  123   0xd5f4b  2      OPC=popq_r64_1        
  popq %r14                                    #  124   0xd5f4d  2      OPC=popq_r64_1        
  popq %r11                                    #  125   0xd5f4f  2      OPC=popq_r64_1        
  andl $0xffffffe0, %r11d                      #  126   0xd5f51  7      OPC=andl_r32_imm32    
  nop                                          #  127   0xd5f58  1      OPC=nop               
  nop                                          #  128   0xd5f59  1      OPC=nop               
  nop                                          #  129   0xd5f5a  1      OPC=nop               
  nop                                          #  130   0xd5f5b  1      OPC=nop               
  addq %r15, %r11                              #  131   0xd5f5c  3      OPC=addq_r64_r64      
  jmpq %r11                                    #  132   0xd5f5f  3      OPC=jmpq_r64          
  nop                                          #  133   0xd5f62  1      OPC=nop               
  nop                                          #  134   0xd5f63  1      OPC=nop               
  nop                                          #  135   0xd5f64  1      OPC=nop               
  nop                                          #  136   0xd5f65  1      OPC=nop               
  nop                                          #  137   0xd5f66  1      OPC=nop               
.L_d5f60:                                      #        0xd5f67  0      OPC=<label>           
  cmpl %r8d, %edx                              #  138   0xd5f67  3      OPC=cmpl_r32_r32      
  jbe .L_d5f40                                 #  139   0xd5f6a  2      OPC=jbe_label         
  nop                                          #  140   0xd5f6c  1      OPC=nop               
  nop                                          #  141   0xd5f6d  1      OPC=nop               
  nop                                          #  142   0xd5f6e  1      OPC=nop               
  nop                                          #  143   0xd5f6f  1      OPC=nop               
  nop                                          #  144   0xd5f70  1      OPC=nop               
  nop                                          #  145   0xd5f71  1      OPC=nop               
  nop                                          #  146   0xd5f72  1      OPC=nop               
  nop                                          #  147   0xd5f73  1      OPC=nop               
  nop                                          #  148   0xd5f74  1      OPC=nop               
  nop                                          #  149   0xd5f75  1      OPC=nop               
  nop                                          #  150   0xd5f76  1      OPC=nop               
  nop                                          #  151   0xd5f77  1      OPC=nop               
  nop                                          #  152   0xd5f78  1      OPC=nop               
  nop                                          #  153   0xd5f79  1      OPC=nop               
  nop                                          #  154   0xd5f7a  1      OPC=nop               
  nop                                          #  155   0xd5f7b  1      OPC=nop               
  nop                                          #  156   0xd5f7c  1      OPC=nop               
  nop                                          #  157   0xd5f7d  1      OPC=nop               
  nop                                          #  158   0xd5f7e  1      OPC=nop               
  nop                                          #  159   0xd5f7f  1      OPC=nop               
  nop                                          #  160   0xd5f80  1      OPC=nop               
  nop                                          #  161   0xd5f81  1      OPC=nop               
  nop                                          #  162   0xd5f82  1      OPC=nop               
  nop                                          #  163   0xd5f83  1      OPC=nop               
  nop                                          #  164   0xd5f84  1      OPC=nop               
  nop                                          #  165   0xd5f85  1      OPC=nop               
  nop                                          #  166   0xd5f86  1      OPC=nop               
.L_d5f80:                                      #        0xd5f87  0      OPC=<label>           
  movl $0xffffffff, %ebx                       #  167   0xd5f87  6      OPC=movl_r32_imm32_1  
  jmpq .L_d5f40                                #  168   0xd5f8d  2      OPC=jmpq_label        
  nop                                          #  169   0xd5f8f  1      OPC=nop               
  nop                                          #  170   0xd5f90  1      OPC=nop               
  nop                                          #  171   0xd5f91  1      OPC=nop               
  nop                                          #  172   0xd5f92  1      OPC=nop               
  nop                                          #  173   0xd5f93  1      OPC=nop               
  nop                                          #  174   0xd5f94  1      OPC=nop               
  nop                                          #  175   0xd5f95  1      OPC=nop               
  nop                                          #  176   0xd5f96  1      OPC=nop               
  nop                                          #  177   0xd5f97  1      OPC=nop               
  nop                                          #  178   0xd5f98  1      OPC=nop               
  nop                                          #  179   0xd5f99  1      OPC=nop               
  nop                                          #  180   0xd5f9a  1      OPC=nop               
  nop                                          #  181   0xd5f9b  1      OPC=nop               
  nop                                          #  182   0xd5f9c  1      OPC=nop               
  nop                                          #  183   0xd5f9d  1      OPC=nop               
  nop                                          #  184   0xd5f9e  1      OPC=nop               
  nop                                          #  185   0xd5f9f  1      OPC=nop               
  nop                                          #  186   0xd5fa0  1      OPC=nop               
  nop                                          #  187   0xd5fa1  1      OPC=nop               
  nop                                          #  188   0xd5fa2  1      OPC=nop               
  nop                                          #  189   0xd5fa3  1      OPC=nop               
  nop                                          #  190   0xd5fa4  1      OPC=nop               
  nop                                          #  191   0xd5fa5  1      OPC=nop               
  nop                                          #  192   0xd5fa6  1      OPC=nop               
  nop                                          #  193   0xd5fa7  1      OPC=nop               
                                                                                              
.size _ZNKSbIwSt11char_traitsIwESaIwEE4findEPKwjj, .-_ZNKSbIwSt11char_traitsIwESaIwEE4findEPKwjj

