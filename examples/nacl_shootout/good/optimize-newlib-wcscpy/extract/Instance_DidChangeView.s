  .text
  .globl Instance_DidChangeView
  .type Instance_DidChangeView, @function

#! file-offset 0x6c340
#! rip-offset  0x2c340
#! capacity    32 bytes

# Text                     #  Line  RIP      Bytes  Opcode              
.Instance_DidChangeView:   #        0x2c340  0      OPC=<label>         
  popq %r11                #  1     0x2c340  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d  #  2     0x2c342  7      OPC=andl_r32_imm32  
  nop                      #  3     0x2c349  1      OPC=nop             
  nop                      #  4     0x2c34a  1      OPC=nop             
  nop                      #  5     0x2c34b  1      OPC=nop             
  nop                      #  6     0x2c34c  1      OPC=nop             
  addq %r15, %r11          #  7     0x2c34d  3      OPC=addq_r64_r64    
  jmpq %r11                #  8     0x2c350  3      OPC=jmpq_r64        
  nop                      #  9     0x2c353  1      OPC=nop             
  nop                      #  10    0x2c354  1      OPC=nop             
  nop                      #  11    0x2c355  1      OPC=nop             
  nop                      #  12    0x2c356  1      OPC=nop             
  nop                      #  13    0x2c357  1      OPC=nop             
  nop                      #  14    0x2c358  1      OPC=nop             
  nop                      #  15    0x2c359  1      OPC=nop             
  nop                      #  16    0x2c35a  1      OPC=nop             
  nop                      #  17    0x2c35b  1      OPC=nop             
  nop                      #  18    0x2c35c  1      OPC=nop             
  nop                      #  19    0x2c35d  1      OPC=nop             
  nop                      #  20    0x2c35e  1      OPC=nop             
  nop                      #  21    0x2c35f  1      OPC=nop             
  nop                      #  22    0x2c360  1      OPC=nop             
  nop                      #  23    0x2c361  1      OPC=nop             
  nop                      #  24    0x2c362  1      OPC=nop             
  nop                      #  25    0x2c363  1      OPC=nop             
  nop                      #  26    0x2c364  1      OPC=nop             
  nop                      #  27    0x2c365  1      OPC=nop             
  nop                      #  28    0x2c366  1      OPC=nop             
                                                                        
.size Instance_DidChangeView, .-Instance_DidChangeView

