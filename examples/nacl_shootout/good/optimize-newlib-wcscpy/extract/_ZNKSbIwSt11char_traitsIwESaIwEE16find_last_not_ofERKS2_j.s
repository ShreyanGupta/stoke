  .text
  .globl _ZNKSbIwSt11char_traitsIwESaIwEE16find_last_not_ofERKS2_j
  .type _ZNKSbIwSt11char_traitsIwESaIwEE16find_last_not_ofERKS2_j, @function

#! file-offset 0x1161c0
#! rip-offset  0xd61c0
#! capacity    32 bytes

# Text                                                            #  Line  RIP      Bytes  Opcode            
._ZNKSbIwSt11char_traitsIwESaIwEE16find_last_not_ofERKS2_j:       #        0xd61c0  0      OPC=<label>       
  movl %esi, %esi                                                 #  1     0xd61c0  2      OPC=movl_r32_r32  
  movl %edi, %edi                                                 #  2     0xd61c2  2      OPC=movl_r32_r32  
  movl %esi, %esi                                                 #  3     0xd61c4  2      OPC=movl_r32_r32  
  movl (%r15,%rsi,1), %esi                                        #  4     0xd61c6  4      OPC=movl_r32_m32  
  leal -0xc(%rsi), %eax                                           #  5     0xd61ca  3      OPC=leal_r32_m16  
  movl %eax, %eax                                                 #  6     0xd61cd  2      OPC=movl_r32_r32  
  movl (%r15,%rax,1), %ecx                                        #  7     0xd61cf  4      OPC=movl_r32_m32  
  jmpq ._ZNKSbIwSt11char_traitsIwESaIwEE16find_last_not_ofEPKwjj  #  8     0xd61d3  5      OPC=jmpq_label_1  
  nop                                                             #  9     0xd61d8  1      OPC=nop           
  nop                                                             #  10    0xd61d9  1      OPC=nop           
  nop                                                             #  11    0xd61da  1      OPC=nop           
  nop                                                             #  12    0xd61db  1      OPC=nop           
  nop                                                             #  13    0xd61dc  1      OPC=nop           
  nop                                                             #  14    0xd61dd  1      OPC=nop           
  nop                                                             #  15    0xd61de  1      OPC=nop           
  nop                                                             #  16    0xd61df  1      OPC=nop           
                                                                                                             
.size _ZNKSbIwSt11char_traitsIwESaIwEE16find_last_not_ofERKS2_j, .-_ZNKSbIwSt11char_traitsIwESaIwEE16find_last_not_ofERKS2_j

