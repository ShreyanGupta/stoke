  .text
  .globl _ZNKSt10moneypunctIcLb0EE10neg_formatEv
  .type _ZNKSt10moneypunctIcLb0EE10neg_formatEv, @function

#! file-offset 0xb6b00
#! rip-offset  0x76b00
#! capacity    64 bytes

# Text                                     #  Line  RIP      Bytes  Opcode              
._ZNKSt10moneypunctIcLb0EE10neg_formatEv:  #        0x76b00  0      OPC=<label>         
  movl %edi, %edi                          #  1     0x76b00  2      OPC=movl_r32_r32    
  subl $0x8, %esp                          #  2     0x76b02  3      OPC=subl_r32_imm8   
  addq %r15, %rsp                          #  3     0x76b05  3      OPC=addq_r64_r64    
  movl %edi, %edi                          #  4     0x76b08  2      OPC=movl_r32_r32    
  movl (%r15,%rdi,1), %eax                 #  5     0x76b0a  4      OPC=movl_r32_m32    
  movl %eax, %eax                          #  6     0x76b0e  2      OPC=movl_r32_r32    
  movl 0x28(%r15,%rax,1), %eax             #  7     0x76b10  5      OPC=movl_r32_m32    
  nop                                      #  8     0x76b15  1      OPC=nop             
  nop                                      #  9     0x76b16  1      OPC=nop             
  nop                                      #  10    0x76b17  1      OPC=nop             
  andl $0xffffffe0, %eax                   #  11    0x76b18  6      OPC=andl_r32_imm32  
  nop                                      #  12    0x76b1e  1      OPC=nop             
  nop                                      #  13    0x76b1f  1      OPC=nop             
  nop                                      #  14    0x76b20  1      OPC=nop             
  addq %r15, %rax                          #  15    0x76b21  3      OPC=addq_r64_r64    
  callq %rax                               #  16    0x76b24  2      OPC=callq_r64       
  addl $0x8, %esp                          #  17    0x76b26  3      OPC=addl_r32_imm8   
  addq %r15, %rsp                          #  18    0x76b29  3      OPC=addq_r64_r64    
  popq %r11                                #  19    0x76b2c  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d                  #  20    0x76b2e  7      OPC=andl_r32_imm32  
  nop                                      #  21    0x76b35  1      OPC=nop             
  nop                                      #  22    0x76b36  1      OPC=nop             
  nop                                      #  23    0x76b37  1      OPC=nop             
  nop                                      #  24    0x76b38  1      OPC=nop             
  addq %r15, %r11                          #  25    0x76b39  3      OPC=addq_r64_r64    
  jmpq %r11                                #  26    0x76b3c  3      OPC=jmpq_r64        
  nop                                      #  27    0x76b3f  1      OPC=nop             
  nop                                      #  28    0x76b40  1      OPC=nop             
  nop                                      #  29    0x76b41  1      OPC=nop             
  nop                                      #  30    0x76b42  1      OPC=nop             
  nop                                      #  31    0x76b43  1      OPC=nop             
  nop                                      #  32    0x76b44  1      OPC=nop             
  nop                                      #  33    0x76b45  1      OPC=nop             
  nop                                      #  34    0x76b46  1      OPC=nop             
  nop                                      #  35    0x76b47  1      OPC=nop             
  nop                                      #  36    0x76b48  1      OPC=nop             
  nop                                      #  37    0x76b49  1      OPC=nop             
  nop                                      #  38    0x76b4a  1      OPC=nop             
  nop                                      #  39    0x76b4b  1      OPC=nop             
  nop                                      #  40    0x76b4c  1      OPC=nop             
                                                                                        
.size _ZNKSt10moneypunctIcLb0EE10neg_formatEv, .-_ZNKSt10moneypunctIcLb0EE10neg_formatEv

