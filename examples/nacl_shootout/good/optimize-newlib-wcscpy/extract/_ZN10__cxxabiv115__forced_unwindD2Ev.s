  .text
  .globl _ZN10__cxxabiv115__forced_unwindD2Ev
  .type _ZN10__cxxabiv115__forced_unwindD2Ev, @function

#! file-offset 0x123200
#! rip-offset  0xe3200
#! capacity    32 bytes

# Text                                  #  Line  RIP      Bytes  Opcode              
._ZN10__cxxabiv115__forced_unwindD2Ev:  #        0xe3200  0      OPC=<label>         
  popq %r11                             #  1     0xe3200  2      OPC=popq_r64_1      
  movl %edi, %edi                       #  2     0xe3202  2      OPC=movl_r32_r32    
  movl %edi, %edi                       #  3     0xe3204  2      OPC=movl_r32_r32    
  movl $0x1003d328, (%r15,%rdi,1)       #  4     0xe3206  8      OPC=movl_m32_imm32  
  andl $0xffffffe0, %r11d               #  5     0xe320e  7      OPC=andl_r32_imm32  
  nop                                   #  6     0xe3215  1      OPC=nop             
  nop                                   #  7     0xe3216  1      OPC=nop             
  nop                                   #  8     0xe3217  1      OPC=nop             
  nop                                   #  9     0xe3218  1      OPC=nop             
  addq %r15, %r11                       #  10    0xe3219  3      OPC=addq_r64_r64    
  jmpq %r11                             #  11    0xe321c  3      OPC=jmpq_r64        
  nop                                   #  12    0xe321f  1      OPC=nop             
  nop                                   #  13    0xe3220  1      OPC=nop             
  nop                                   #  14    0xe3221  1      OPC=nop             
  nop                                   #  15    0xe3222  1      OPC=nop             
  nop                                   #  16    0xe3223  1      OPC=nop             
  nop                                   #  17    0xe3224  1      OPC=nop             
  nop                                   #  18    0xe3225  1      OPC=nop             
  nop                                   #  19    0xe3226  1      OPC=nop             
                                                                                     
.size _ZN10__cxxabiv115__forced_unwindD2Ev, .-_ZN10__cxxabiv115__forced_unwindD2Ev

