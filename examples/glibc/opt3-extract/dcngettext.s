  .text
  .globl dcngettext
  .type dcngettext, @function

#! file-offset 0x2fee0
#! rip-offset  0x2fee0
#! capacity    16 bytes

# Text                #  Line  RIP      Bytes  Opcode              
.dcngettext:          #        0x2fee0  0      OPC=<label>         
  movl %r8d, %r9d     #  1     0x2fee0  3      OPC=movl_r32_r32    
  movq %rcx, %r8      #  2     0x2fee3  3      OPC=movq_r64_r64    
  movl $0x1, %ecx     #  3     0x2fee6  5      OPC=movl_r32_imm32  
  jmpq .__dcigettext  #  4     0x2feeb  5      OPC=jmpq_label_1    
                                                                   
.size dcngettext, .-dcngettext
