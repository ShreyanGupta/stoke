  .text
  .globl _ZNSt15basic_streambufIwSt11char_traitsIwEE6snextcEv
  .type _ZNSt15basic_streambufIwSt11char_traitsIwEE6snextcEv, @function

#! file-offset 0xe9aa0
#! rip-offset  0xa9aa0
#! capacity    224 bytes

# Text                                                  #  Line  RIP      Bytes  Opcode              
._ZNSt15basic_streambufIwSt11char_traitsIwEE6snextcEv:  #        0xa9aa0  0      OPC=<label>         
  pushq %rbx                                            #  1     0xa9aa0  1      OPC=pushq_r64_1     
  movl %edi, %ebx                                       #  2     0xa9aa1  2      OPC=movl_r32_r32    
  movl %ebx, %ebx                                       #  3     0xa9aa3  2      OPC=movl_r32_r32    
  movl 0x8(%r15,%rbx,1), %edx                           #  4     0xa9aa5  5      OPC=movl_r32_m32    
  movl %ebx, %ebx                                       #  5     0xa9aaa  2      OPC=movl_r32_r32    
  cmpl %edx, 0xc(%r15,%rbx,1)                           #  6     0xa9aac  5      OPC=cmpl_m32_r32    
  jbe .L_a9b20                                          #  7     0xa9ab1  2      OPC=jbe_label       
  movl %edx, %edx                                       #  8     0xa9ab3  2      OPC=movl_r32_r32    
  movl (%r15,%rdx,1), %eax                              #  9     0xa9ab5  4      OPC=movl_r32_m32    
  addl $0x4, %edx                                       #  10    0xa9ab9  3      OPC=addl_r32_imm8   
  nop                                                   #  11    0xa9abc  1      OPC=nop             
  nop                                                   #  12    0xa9abd  1      OPC=nop             
  nop                                                   #  13    0xa9abe  1      OPC=nop             
  nop                                                   #  14    0xa9abf  1      OPC=nop             
  movl %ebx, %ebx                                       #  15    0xa9ac0  2      OPC=movl_r32_r32    
  movl %edx, 0x8(%r15,%rbx,1)                           #  16    0xa9ac2  5      OPC=movl_m32_r32    
  nop                                                   #  17    0xa9ac7  1      OPC=nop             
  nop                                                   #  18    0xa9ac8  1      OPC=nop             
  nop                                                   #  19    0xa9ac9  1      OPC=nop             
  nop                                                   #  20    0xa9aca  1      OPC=nop             
  nop                                                   #  21    0xa9acb  1      OPC=nop             
  nop                                                   #  22    0xa9acc  1      OPC=nop             
  nop                                                   #  23    0xa9acd  1      OPC=nop             
  nop                                                   #  24    0xa9ace  1      OPC=nop             
  nop                                                   #  25    0xa9acf  1      OPC=nop             
  nop                                                   #  26    0xa9ad0  1      OPC=nop             
  nop                                                   #  27    0xa9ad1  1      OPC=nop             
  nop                                                   #  28    0xa9ad2  1      OPC=nop             
  nop                                                   #  29    0xa9ad3  1      OPC=nop             
  nop                                                   #  30    0xa9ad4  1      OPC=nop             
  nop                                                   #  31    0xa9ad5  1      OPC=nop             
  nop                                                   #  32    0xa9ad6  1      OPC=nop             
  nop                                                   #  33    0xa9ad7  1      OPC=nop             
  nop                                                   #  34    0xa9ad8  1      OPC=nop             
  nop                                                   #  35    0xa9ad9  1      OPC=nop             
  nop                                                   #  36    0xa9ada  1      OPC=nop             
  nop                                                   #  37    0xa9adb  1      OPC=nop             
  nop                                                   #  38    0xa9adc  1      OPC=nop             
  nop                                                   #  39    0xa9add  1      OPC=nop             
  nop                                                   #  40    0xa9ade  1      OPC=nop             
  nop                                                   #  41    0xa9adf  1      OPC=nop             
.L_a9ae0:                                               #        0xa9ae0  0      OPC=<label>         
  cmpl $0xffffffff, %eax                                #  42    0xa9ae0  6      OPC=cmpl_r32_imm32  
  nop                                                   #  43    0xa9ae6  1      OPC=nop             
  nop                                                   #  44    0xa9ae7  1      OPC=nop             
  nop                                                   #  45    0xa9ae8  1      OPC=nop             
  je .L_a9b00                                           #  46    0xa9ae9  2      OPC=je_label        
  movl %ebx, %ebx                                       #  47    0xa9aeb  2      OPC=movl_r32_r32    
  movl 0x8(%r15,%rbx,1), %eax                           #  48    0xa9aed  5      OPC=movl_r32_m32    
  movl %ebx, %ebx                                       #  49    0xa9af2  2      OPC=movl_r32_r32    
  cmpl %eax, 0xc(%r15,%rbx,1)                           #  50    0xa9af4  5      OPC=cmpl_m32_r32    
  jbe .L_a9b60                                          #  51    0xa9af9  2      OPC=jbe_label       
  movl %eax, %eax                                       #  52    0xa9afb  2      OPC=movl_r32_r32    
  movl (%r15,%rax,1), %eax                              #  53    0xa9afd  4      OPC=movl_r32_m32    
  nop                                                   #  54    0xa9b01  1      OPC=nop             
  nop                                                   #  55    0xa9b02  1      OPC=nop             
  nop                                                   #  56    0xa9b03  1      OPC=nop             
  nop                                                   #  57    0xa9b04  1      OPC=nop             
  nop                                                   #  58    0xa9b05  1      OPC=nop             
.L_a9b00:                                               #        0xa9b06  0      OPC=<label>         
  popq %rbx                                             #  59    0xa9b06  1      OPC=popq_r64_1      
  popq %r11                                             #  60    0xa9b07  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d                               #  61    0xa9b09  7      OPC=andl_r32_imm32  
  nop                                                   #  62    0xa9b10  1      OPC=nop             
  nop                                                   #  63    0xa9b11  1      OPC=nop             
  nop                                                   #  64    0xa9b12  1      OPC=nop             
  nop                                                   #  65    0xa9b13  1      OPC=nop             
  addq %r15, %r11                                       #  66    0xa9b14  3      OPC=addq_r64_r64    
  jmpq %r11                                             #  67    0xa9b17  3      OPC=jmpq_r64        
  nop                                                   #  68    0xa9b1a  1      OPC=nop             
  nop                                                   #  69    0xa9b1b  1      OPC=nop             
  nop                                                   #  70    0xa9b1c  1      OPC=nop             
  nop                                                   #  71    0xa9b1d  1      OPC=nop             
  nop                                                   #  72    0xa9b1e  1      OPC=nop             
  nop                                                   #  73    0xa9b1f  1      OPC=nop             
  nop                                                   #  74    0xa9b20  1      OPC=nop             
  nop                                                   #  75    0xa9b21  1      OPC=nop             
  nop                                                   #  76    0xa9b22  1      OPC=nop             
  nop                                                   #  77    0xa9b23  1      OPC=nop             
  nop                                                   #  78    0xa9b24  1      OPC=nop             
  nop                                                   #  79    0xa9b25  1      OPC=nop             
  nop                                                   #  80    0xa9b26  1      OPC=nop             
  nop                                                   #  81    0xa9b27  1      OPC=nop             
  nop                                                   #  82    0xa9b28  1      OPC=nop             
  nop                                                   #  83    0xa9b29  1      OPC=nop             
  nop                                                   #  84    0xa9b2a  1      OPC=nop             
  nop                                                   #  85    0xa9b2b  1      OPC=nop             
  nop                                                   #  86    0xa9b2c  1      OPC=nop             
.L_a9b20:                                               #        0xa9b2d  0      OPC=<label>         
  movl %ebx, %ebx                                       #  87    0xa9b2d  2      OPC=movl_r32_r32    
  movl (%r15,%rbx,1), %eax                              #  88    0xa9b2f  4      OPC=movl_r32_m32    
  movl %ebx, %edi                                       #  89    0xa9b33  2      OPC=movl_r32_r32    
  movl %eax, %eax                                       #  90    0xa9b35  2      OPC=movl_r32_r32    
  movl 0x28(%r15,%rax,1), %eax                          #  91    0xa9b37  5      OPC=movl_r32_m32    
  nop                                                   #  92    0xa9b3c  1      OPC=nop             
  nop                                                   #  93    0xa9b3d  1      OPC=nop             
  nop                                                   #  94    0xa9b3e  1      OPC=nop             
  nop                                                   #  95    0xa9b3f  1      OPC=nop             
  nop                                                   #  96    0xa9b40  1      OPC=nop             
  nop                                                   #  97    0xa9b41  1      OPC=nop             
  nop                                                   #  98    0xa9b42  1      OPC=nop             
  nop                                                   #  99    0xa9b43  1      OPC=nop             
  nop                                                   #  100   0xa9b44  1      OPC=nop             
  andl $0xffffffe0, %eax                                #  101   0xa9b45  6      OPC=andl_r32_imm32  
  nop                                                   #  102   0xa9b4b  1      OPC=nop             
  nop                                                   #  103   0xa9b4c  1      OPC=nop             
  nop                                                   #  104   0xa9b4d  1      OPC=nop             
  addq %r15, %rax                                       #  105   0xa9b4e  3      OPC=addq_r64_r64    
  callq %rax                                            #  106   0xa9b51  2      OPC=callq_r64       
  jmpq .L_a9ae0                                         #  107   0xa9b53  2      OPC=jmpq_label      
  nop                                                   #  108   0xa9b55  1      OPC=nop             
  nop                                                   #  109   0xa9b56  1      OPC=nop             
  nop                                                   #  110   0xa9b57  1      OPC=nop             
  nop                                                   #  111   0xa9b58  1      OPC=nop             
  nop                                                   #  112   0xa9b59  1      OPC=nop             
  nop                                                   #  113   0xa9b5a  1      OPC=nop             
  nop                                                   #  114   0xa9b5b  1      OPC=nop             
  nop                                                   #  115   0xa9b5c  1      OPC=nop             
  nop                                                   #  116   0xa9b5d  1      OPC=nop             
  nop                                                   #  117   0xa9b5e  1      OPC=nop             
  nop                                                   #  118   0xa9b5f  1      OPC=nop             
  nop                                                   #  119   0xa9b60  1      OPC=nop             
  nop                                                   #  120   0xa9b61  1      OPC=nop             
  nop                                                   #  121   0xa9b62  1      OPC=nop             
  nop                                                   #  122   0xa9b63  1      OPC=nop             
  nop                                                   #  123   0xa9b64  1      OPC=nop             
  nop                                                   #  124   0xa9b65  1      OPC=nop             
  nop                                                   #  125   0xa9b66  1      OPC=nop             
  nop                                                   #  126   0xa9b67  1      OPC=nop             
  nop                                                   #  127   0xa9b68  1      OPC=nop             
  nop                                                   #  128   0xa9b69  1      OPC=nop             
  nop                                                   #  129   0xa9b6a  1      OPC=nop             
  nop                                                   #  130   0xa9b6b  1      OPC=nop             
  nop                                                   #  131   0xa9b6c  1      OPC=nop             
  nop                                                   #  132   0xa9b6d  1      OPC=nop             
  nop                                                   #  133   0xa9b6e  1      OPC=nop             
  nop                                                   #  134   0xa9b6f  1      OPC=nop             
  nop                                                   #  135   0xa9b70  1      OPC=nop             
  nop                                                   #  136   0xa9b71  1      OPC=nop             
  nop                                                   #  137   0xa9b72  1      OPC=nop             
.L_a9b60:                                               #        0xa9b73  0      OPC=<label>         
  movl %ebx, %ebx                                       #  138   0xa9b73  2      OPC=movl_r32_r32    
  movl (%r15,%rbx,1), %eax                              #  139   0xa9b75  4      OPC=movl_r32_m32    
  movl %ebx, %edi                                       #  140   0xa9b79  2      OPC=movl_r32_r32    
  popq %rbx                                             #  141   0xa9b7b  1      OPC=popq_r64_1      
  movl %eax, %eax                                       #  142   0xa9b7c  2      OPC=movl_r32_r32    
  movl 0x24(%r15,%rax,1), %eax                          #  143   0xa9b7e  5      OPC=movl_r32_m32    
  andl $0xffffffe0, %eax                                #  144   0xa9b83  6      OPC=andl_r32_imm32  
  nop                                                   #  145   0xa9b89  1      OPC=nop             
  nop                                                   #  146   0xa9b8a  1      OPC=nop             
  nop                                                   #  147   0xa9b8b  1      OPC=nop             
  addq %r15, %rax                                       #  148   0xa9b8c  3      OPC=addq_r64_r64    
  jmpq %rax                                             #  149   0xa9b8f  2      OPC=jmpq_r64        
  nop                                                   #  150   0xa9b91  1      OPC=nop             
  nop                                                   #  151   0xa9b92  1      OPC=nop             
  nop                                                   #  152   0xa9b93  1      OPC=nop             
  nop                                                   #  153   0xa9b94  1      OPC=nop             
  nop                                                   #  154   0xa9b95  1      OPC=nop             
  nop                                                   #  155   0xa9b96  1      OPC=nop             
  nop                                                   #  156   0xa9b97  1      OPC=nop             
  nop                                                   #  157   0xa9b98  1      OPC=nop             
                                                                                                     
.size _ZNSt15basic_streambufIwSt11char_traitsIwEE6snextcEv, .-_ZNSt15basic_streambufIwSt11char_traitsIwEE6snextcEv

