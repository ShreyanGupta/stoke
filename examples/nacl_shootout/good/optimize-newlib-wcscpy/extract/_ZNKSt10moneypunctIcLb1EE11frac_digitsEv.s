  .text
  .globl _ZNKSt10moneypunctIcLb1EE11frac_digitsEv
  .type _ZNKSt10moneypunctIcLb1EE11frac_digitsEv, @function

#! file-offset 0xb6d20
#! rip-offset  0x76d20
#! capacity    32 bytes

# Text                                      #  Line  RIP      Bytes  Opcode              
._ZNKSt10moneypunctIcLb1EE11frac_digitsEv:  #        0x76d20  0      OPC=<label>         
  movl %edi, %edi                           #  1     0x76d20  2      OPC=movl_r32_r32    
  movl %edi, %edi                           #  2     0x76d22  2      OPC=movl_r32_r32    
  movl (%r15,%rdi,1), %eax                  #  3     0x76d24  4      OPC=movl_r32_m32    
  movl %eax, %eax                           #  4     0x76d28  2      OPC=movl_r32_r32    
  movl 0x20(%r15,%rax,1), %eax              #  5     0x76d2a  5      OPC=movl_r32_m32    
  andl $0xffffffe0, %eax                    #  6     0x76d2f  6      OPC=andl_r32_imm32  
  nop                                       #  7     0x76d35  1      OPC=nop             
  nop                                       #  8     0x76d36  1      OPC=nop             
  nop                                       #  9     0x76d37  1      OPC=nop             
  addq %r15, %rax                           #  10    0x76d38  3      OPC=addq_r64_r64    
  jmpq %rax                                 #  11    0x76d3b  2      OPC=jmpq_r64        
  nop                                       #  12    0x76d3d  1      OPC=nop             
  nop                                       #  13    0x76d3e  1      OPC=nop             
  nop                                       #  14    0x76d3f  1      OPC=nop             
  nop                                       #  15    0x76d40  1      OPC=nop             
  nop                                       #  16    0x76d41  1      OPC=nop             
  nop                                       #  17    0x76d42  1      OPC=nop             
  nop                                       #  18    0x76d43  1      OPC=nop             
  nop                                       #  19    0x76d44  1      OPC=nop             
  nop                                       #  20    0x76d45  1      OPC=nop             
                                                                                         
.size _ZNKSt10moneypunctIcLb1EE11frac_digitsEv, .-_ZNKSt10moneypunctIcLb1EE11frac_digitsEv

