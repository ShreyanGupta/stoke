  .text
  .globl _ZNKSs5rfindERKSsj
  .type _ZNKSs5rfindERKSsj, @function

#! file-offset 0xeb4c0
#! rip-offset  0xab4c0
#! capacity    32 bytes

# Text                      #  Line  RIP      Bytes  Opcode            
._ZNKSs5rfindERKSsj:        #        0xab4c0  0      OPC=<label>       
  movl %esi, %esi           #  1     0xab4c0  2      OPC=movl_r32_r32  
  movl %edi, %edi           #  2     0xab4c2  2      OPC=movl_r32_r32  
  movl %esi, %esi           #  3     0xab4c4  2      OPC=movl_r32_r32  
  movl (%r15,%rsi,1), %esi  #  4     0xab4c6  4      OPC=movl_r32_m32  
  leal -0xc(%rsi), %eax     #  5     0xab4ca  3      OPC=leal_r32_m16  
  movl %eax, %eax           #  6     0xab4cd  2      OPC=movl_r32_r32  
  movl (%r15,%rax,1), %ecx  #  7     0xab4cf  4      OPC=movl_r32_m32  
  jmpq ._ZNKSs5rfindEPKcjj  #  8     0xab4d3  5      OPC=jmpq_label_1  
  nop                       #  9     0xab4d8  1      OPC=nop           
  nop                       #  10    0xab4d9  1      OPC=nop           
  nop                       #  11    0xab4da  1      OPC=nop           
  nop                       #  12    0xab4db  1      OPC=nop           
  nop                       #  13    0xab4dc  1      OPC=nop           
  nop                       #  14    0xab4dd  1      OPC=nop           
  nop                       #  15    0xab4de  1      OPC=nop           
  nop                       #  16    0xab4df  1      OPC=nop           
                                                                       
.size _ZNKSs5rfindERKSsj, .-_ZNKSs5rfindERKSsj

