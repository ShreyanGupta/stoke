  .text
  .globl _ZNSbIwSt11char_traitsIwESaIwEE6appendEPKw
  .type _ZNSbIwSt11char_traitsIwESaIwEE6appendEPKw, @function

#! file-offset 0x1186e0
#! rip-offset  0xd86e0
#! capacity    64 bytes

# Text                                               #  Line  RIP      Bytes  Opcode             
._ZNSbIwSt11char_traitsIwESaIwEE6appendEPKw:         #        0xd86e0  0      OPC=<label>        
  movq %r12, -0x8(%rsp)                              #  1     0xd86e0  5      OPC=movq_m64_r64   
  movl %esi, %r12d                                   #  2     0xd86e5  3      OPC=movl_r32_r32   
  movq %rbx, -0x10(%rsp)                             #  3     0xd86e8  5      OPC=movq_m64_r64   
  subl $0x18, %esp                                   #  4     0xd86ed  3      OPC=subl_r32_imm8  
  addq %r15, %rsp                                    #  5     0xd86f0  3      OPC=addq_r64_r64   
  movl %edi, %ebx                                    #  6     0xd86f3  2      OPC=movl_r32_r32   
  movl %r12d, %edi                                   #  7     0xd86f5  3      OPC=movl_r32_r32   
  nop                                                #  8     0xd86f8  1      OPC=nop            
  nop                                                #  9     0xd86f9  1      OPC=nop            
  nop                                                #  10    0xd86fa  1      OPC=nop            
  callq .wcslen                                      #  11    0xd86fb  5      OPC=callq_label    
  movl %r12d, %esi                                   #  12    0xd8700  3      OPC=movl_r32_r32   
  movl %ebx, %edi                                    #  13    0xd8703  2      OPC=movl_r32_r32   
  movq 0x10(%rsp), %r12                              #  14    0xd8705  5      OPC=movq_r64_m64   
  movq 0x8(%rsp), %rbx                               #  15    0xd870a  5      OPC=movq_r64_m64   
  movl %eax, %edx                                    #  16    0xd870f  2      OPC=movl_r32_r32   
  addl $0x18, %esp                                   #  17    0xd8711  3      OPC=addl_r32_imm8  
  addq %r15, %rsp                                    #  18    0xd8714  3      OPC=addq_r64_r64   
  jmpq ._ZNSbIwSt11char_traitsIwESaIwEE6appendEPKwj  #  19    0xd8717  5      OPC=jmpq_label_1   
  nop                                                #  20    0xd871c  1      OPC=nop            
  nop                                                #  21    0xd871d  1      OPC=nop            
  nop                                                #  22    0xd871e  1      OPC=nop            
  nop                                                #  23    0xd871f  1      OPC=nop            
                                                                                                 
.size _ZNSbIwSt11char_traitsIwESaIwEE6appendEPKw, .-_ZNSbIwSt11char_traitsIwESaIwEE6appendEPKw

