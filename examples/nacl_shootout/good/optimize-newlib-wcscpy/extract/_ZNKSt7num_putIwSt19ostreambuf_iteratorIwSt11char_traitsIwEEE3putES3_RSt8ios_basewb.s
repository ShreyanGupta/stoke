  .text
  .globl _ZNKSt7num_putIwSt19ostreambuf_iteratorIwSt11char_traitsIwEEE3putES3_RSt8ios_basewb
  .type _ZNKSt7num_putIwSt19ostreambuf_iteratorIwSt11char_traitsIwEEE3putES3_RSt8ios_basewb, @function

#! file-offset 0xf0c20
#! rip-offset  0xb0c20
#! capacity    96 bytes

# Text                                                                                 #  Line  RIP      Bytes  Opcode              
._ZNKSt7num_putIwSt19ostreambuf_iteratorIwSt11char_traitsIwEEE3putES3_RSt8ios_basewb:  #        0xb0c20  0      OPC=<label>         
  movl %edi, %edi                                                                      #  1     0xb0c20  2      OPC=movl_r32_r32    
  subl $0x8, %esp                                                                      #  2     0xb0c22  3      OPC=subl_r32_imm8   
  addq %r15, %rsp                                                                      #  3     0xb0c25  3      OPC=addq_r64_r64    
  movl %edx, %edx                                                                      #  4     0xb0c28  2      OPC=movl_r32_r32    
  movl %edi, %edi                                                                      #  5     0xb0c2a  2      OPC=movl_r32_r32    
  movl (%r15,%rdi,1), %eax                                                             #  6     0xb0c2c  4      OPC=movl_r32_m32    
  movzbl %r8b, %r8d                                                                    #  7     0xb0c30  4      OPC=movzbl_r32_r8   
  movl %eax, %eax                                                                      #  8     0xb0c34  2      OPC=movl_r32_r32    
  movl 0x8(%r15,%rax,1), %eax                                                          #  9     0xb0c36  5      OPC=movl_r32_m32    
  nop                                                                                  #  10    0xb0c3b  1      OPC=nop             
  nop                                                                                  #  11    0xb0c3c  1      OPC=nop             
  nop                                                                                  #  12    0xb0c3d  1      OPC=nop             
  nop                                                                                  #  13    0xb0c3e  1      OPC=nop             
  nop                                                                                  #  14    0xb0c3f  1      OPC=nop             
  nop                                                                                  #  15    0xb0c40  1      OPC=nop             
  nop                                                                                  #  16    0xb0c41  1      OPC=nop             
  nop                                                                                  #  17    0xb0c42  1      OPC=nop             
  nop                                                                                  #  18    0xb0c43  1      OPC=nop             
  nop                                                                                  #  19    0xb0c44  1      OPC=nop             
  nop                                                                                  #  20    0xb0c45  1      OPC=nop             
  nop                                                                                  #  21    0xb0c46  1      OPC=nop             
  nop                                                                                  #  22    0xb0c47  1      OPC=nop             
  nop                                                                                  #  23    0xb0c48  1      OPC=nop             
  nop                                                                                  #  24    0xb0c49  1      OPC=nop             
  nop                                                                                  #  25    0xb0c4a  1      OPC=nop             
  nop                                                                                  #  26    0xb0c4b  1      OPC=nop             
  nop                                                                                  #  27    0xb0c4c  1      OPC=nop             
  nop                                                                                  #  28    0xb0c4d  1      OPC=nop             
  nop                                                                                  #  29    0xb0c4e  1      OPC=nop             
  nop                                                                                  #  30    0xb0c4f  1      OPC=nop             
  nop                                                                                  #  31    0xb0c50  1      OPC=nop             
  nop                                                                                  #  32    0xb0c51  1      OPC=nop             
  nop                                                                                  #  33    0xb0c52  1      OPC=nop             
  nop                                                                                  #  34    0xb0c53  1      OPC=nop             
  nop                                                                                  #  35    0xb0c54  1      OPC=nop             
  nop                                                                                  #  36    0xb0c55  1      OPC=nop             
  nop                                                                                  #  37    0xb0c56  1      OPC=nop             
  nop                                                                                  #  38    0xb0c57  1      OPC=nop             
  andl $0xffffffe0, %eax                                                               #  39    0xb0c58  6      OPC=andl_r32_imm32  
  nop                                                                                  #  40    0xb0c5e  1      OPC=nop             
  nop                                                                                  #  41    0xb0c5f  1      OPC=nop             
  nop                                                                                  #  42    0xb0c60  1      OPC=nop             
  addq %r15, %rax                                                                      #  43    0xb0c61  3      OPC=addq_r64_r64    
  callq %rax                                                                           #  44    0xb0c64  2      OPC=callq_r64       
  addl $0x8, %esp                                                                      #  45    0xb0c66  3      OPC=addl_r32_imm8   
  addq %r15, %rsp                                                                      #  46    0xb0c69  3      OPC=addq_r64_r64    
  popq %r11                                                                            #  47    0xb0c6c  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d                                                              #  48    0xb0c6e  7      OPC=andl_r32_imm32  
  nop                                                                                  #  49    0xb0c75  1      OPC=nop             
  nop                                                                                  #  50    0xb0c76  1      OPC=nop             
  nop                                                                                  #  51    0xb0c77  1      OPC=nop             
  nop                                                                                  #  52    0xb0c78  1      OPC=nop             
  addq %r15, %r11                                                                      #  53    0xb0c79  3      OPC=addq_r64_r64    
  jmpq %r11                                                                            #  54    0xb0c7c  3      OPC=jmpq_r64        
  nop                                                                                  #  55    0xb0c7f  1      OPC=nop             
  nop                                                                                  #  56    0xb0c80  1      OPC=nop             
  nop                                                                                  #  57    0xb0c81  1      OPC=nop             
  nop                                                                                  #  58    0xb0c82  1      OPC=nop             
  nop                                                                                  #  59    0xb0c83  1      OPC=nop             
  nop                                                                                  #  60    0xb0c84  1      OPC=nop             
  nop                                                                                  #  61    0xb0c85  1      OPC=nop             
  nop                                                                                  #  62    0xb0c86  1      OPC=nop             
  nop                                                                                  #  63    0xb0c87  1      OPC=nop             
  nop                                                                                  #  64    0xb0c88  1      OPC=nop             
  nop                                                                                  #  65    0xb0c89  1      OPC=nop             
  nop                                                                                  #  66    0xb0c8a  1      OPC=nop             
  nop                                                                                  #  67    0xb0c8b  1      OPC=nop             
  nop                                                                                  #  68    0xb0c8c  1      OPC=nop             
                                                                                                                                    
.size _ZNKSt7num_putIwSt19ostreambuf_iteratorIwSt11char_traitsIwEEE3putES3_RSt8ios_basewb, .-_ZNKSt7num_putIwSt19ostreambuf_iteratorIwSt11char_traitsIwEEE3putES3_RSt8ios_basewb

