  .text
  .globl labs
  .type labs, @function

#! file-offset 0x189760
#! rip-offset  0x149760
#! capacity    32 bytes

# Text                     #  Line  RIP       Bytes  Opcode              
.labs:                     #        0x149760  0      OPC=<label>         
  movl %edi, %edx          #  1     0x149760  2      OPC=movl_r32_r32    
  popq %r11                #  2     0x149762  2      OPC=popq_r64_1      
  sarl $0x1f, %edx         #  3     0x149764  3      OPC=sarl_r32_imm8   
  movl %edx, %eax          #  4     0x149767  2      OPC=movl_r32_r32    
  xorl %edi, %eax          #  5     0x149769  2      OPC=xorl_r32_r32    
  subl %edx, %eax          #  6     0x14976b  2      OPC=subl_r32_r32    
  andl $0xffffffe0, %r11d  #  7     0x14976d  7      OPC=andl_r32_imm32  
  nop                      #  8     0x149774  1      OPC=nop             
  nop                      #  9     0x149775  1      OPC=nop             
  nop                      #  10    0x149776  1      OPC=nop             
  nop                      #  11    0x149777  1      OPC=nop             
  addq %r15, %r11          #  12    0x149778  3      OPC=addq_r64_r64    
  jmpq %r11                #  13    0x14977b  3      OPC=jmpq_r64        
  nop                      #  14    0x14977e  1      OPC=nop             
  nop                      #  15    0x14977f  1      OPC=nop             
  nop                      #  16    0x149780  1      OPC=nop             
  nop                      #  17    0x149781  1      OPC=nop             
  nop                      #  18    0x149782  1      OPC=nop             
  nop                      #  19    0x149783  1      OPC=nop             
  nop                      #  20    0x149784  1      OPC=nop             
  nop                      #  21    0x149785  1      OPC=nop             
  nop                      #  22    0x149786  1      OPC=nop             
                                                                         
.size labs, .-labs

