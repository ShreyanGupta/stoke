  .text
  .globl tp_add_job
  .type tp_add_job, @function

#! file-offset 0x63420
#! rip-offset  0x23420
#! capacity    96 bytes

# Text                         #  Line  RIP      Bytes  Opcode              
.tp_add_job:                   #        0x23420  0      OPC=<label>         
  movl %edi, %edi              #  1     0x23420  2      OPC=movl_r32_r32    
  movl %esi, %esi              #  2     0x23422  2      OPC=movl_r32_r32    
  movl %edx, %edx              #  3     0x23424  2      OPC=movl_r32_r32    
  movl %edi, %edi              #  4     0x23426  2      OPC=movl_r32_r32    
  movl 0x8(%r15,%rdi,1), %eax  #  5     0x23428  5      OPC=movl_r32_m32    
  movl %edi, %edi              #  6     0x2342d  2      OPC=movl_r32_r32    
  cmpl 0x4(%r15,%rdi,1), %eax  #  7     0x2342f  5      OPC=cmpl_r32_m32    
  jge .L_23460                 #  8     0x23434  2      OPC=jge_label       
  shll $0x3, %eax              #  9     0x23436  3      OPC=shll_r32_imm8   
  movl %edi, %edi              #  10    0x23439  2      OPC=movl_r32_r32    
  addl (%r15,%rdi,1), %eax     #  11    0x2343b  4      OPC=addl_r32_m32    
  nop                          #  12    0x2343f  1      OPC=nop             
  movl %eax, %eax              #  13    0x23440  2      OPC=movl_r32_r32    
  movl %esi, (%r15,%rax,1)     #  14    0x23442  4      OPC=movl_m32_r32    
  movl %eax, %eax              #  15    0x23446  2      OPC=movl_r32_r32    
  movl %edx, 0x4(%r15,%rax,1)  #  16    0x23448  5      OPC=movl_m32_r32    
  movl %edi, %edi              #  17    0x2344d  2      OPC=movl_r32_r32    
  addl $0x1, 0x8(%r15,%rdi,1)  #  18    0x2344f  6      OPC=addl_m32_imm8   
  nop                          #  19    0x23455  1      OPC=nop             
  nop                          #  20    0x23456  1      OPC=nop             
  nop                          #  21    0x23457  1      OPC=nop             
  nop                          #  22    0x23458  1      OPC=nop             
  nop                          #  23    0x23459  1      OPC=nop             
  nop                          #  24    0x2345a  1      OPC=nop             
  nop                          #  25    0x2345b  1      OPC=nop             
  nop                          #  26    0x2345c  1      OPC=nop             
  nop                          #  27    0x2345d  1      OPC=nop             
  nop                          #  28    0x2345e  1      OPC=nop             
  nop                          #  29    0x2345f  1      OPC=nop             
.L_23460:                      #        0x23460  0      OPC=<label>         
  popq %r11                    #  30    0x23460  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d      #  31    0x23462  7      OPC=andl_r32_imm32  
  nop                          #  32    0x23469  1      OPC=nop             
  nop                          #  33    0x2346a  1      OPC=nop             
  nop                          #  34    0x2346b  1      OPC=nop             
  nop                          #  35    0x2346c  1      OPC=nop             
  addq %r15, %r11              #  36    0x2346d  3      OPC=addq_r64_r64    
  jmpq %r11                    #  37    0x23470  3      OPC=jmpq_r64        
  nop                          #  38    0x23473  1      OPC=nop             
  nop                          #  39    0x23474  1      OPC=nop             
  nop                          #  40    0x23475  1      OPC=nop             
  nop                          #  41    0x23476  1      OPC=nop             
  nop                          #  42    0x23477  1      OPC=nop             
  nop                          #  43    0x23478  1      OPC=nop             
  nop                          #  44    0x23479  1      OPC=nop             
  nop                          #  45    0x2347a  1      OPC=nop             
  nop                          #  46    0x2347b  1      OPC=nop             
  nop                          #  47    0x2347c  1      OPC=nop             
  nop                          #  48    0x2347d  1      OPC=nop             
  nop                          #  49    0x2347e  1      OPC=nop             
  nop                          #  50    0x2347f  1      OPC=nop             
  nop                          #  51    0x23480  1      OPC=nop             
  nop                          #  52    0x23481  1      OPC=nop             
  nop                          #  53    0x23482  1      OPC=nop             
  nop                          #  54    0x23483  1      OPC=nop             
  nop                          #  55    0x23484  1      OPC=nop             
  nop                          #  56    0x23485  1      OPC=nop             
  nop                          #  57    0x23486  1      OPC=nop             
                                                                            
.size tp_add_job, .-tp_add_job

