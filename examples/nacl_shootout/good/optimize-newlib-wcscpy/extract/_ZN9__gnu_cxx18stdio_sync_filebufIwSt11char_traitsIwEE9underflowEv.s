  .text
  .globl _ZN9__gnu_cxx18stdio_sync_filebufIwSt11char_traitsIwEE9underflowEv
  .type _ZN9__gnu_cxx18stdio_sync_filebufIwSt11char_traitsIwEE9underflowEv, @function

#! file-offset 0xdbd40
#! rip-offset  0x9bd40
#! capacity    64 bytes

# Text                                                                #  Line  RIP      Bytes  Opcode            
._ZN9__gnu_cxx18stdio_sync_filebufIwSt11char_traitsIwEE9underflowEv:  #        0x9bd40  0      OPC=<label>       
  pushq %rbx                                                          #  1     0x9bd40  1      OPC=pushq_r64_1   
  movl %edi, %ebx                                                     #  2     0x9bd41  2      OPC=movl_r32_r32  
  movl %ebx, %ebx                                                     #  3     0x9bd43  2      OPC=movl_r32_r32  
  movl 0x20(%r15,%rbx,1), %edi                                        #  4     0x9bd45  5      OPC=movl_r32_m32  
  xchgw %ax, %ax                                                      #  5     0x9bd4a  2      OPC=xchgw_ax_r16  
  nop                                                                 #  6     0x9bd4c  1      OPC=nop           
  nop                                                                 #  7     0x9bd4d  1      OPC=nop           
  nop                                                                 #  8     0x9bd4e  1      OPC=nop           
  nop                                                                 #  9     0x9bd4f  1      OPC=nop           
  nop                                                                 #  10    0x9bd50  1      OPC=nop           
  nop                                                                 #  11    0x9bd51  1      OPC=nop           
  nop                                                                 #  12    0x9bd52  1      OPC=nop           
  nop                                                                 #  13    0x9bd53  1      OPC=nop           
  nop                                                                 #  14    0x9bd54  1      OPC=nop           
  nop                                                                 #  15    0x9bd55  1      OPC=nop           
  nop                                                                 #  16    0x9bd56  1      OPC=nop           
  nop                                                                 #  17    0x9bd57  1      OPC=nop           
  nop                                                                 #  18    0x9bd58  1      OPC=nop           
  nop                                                                 #  19    0x9bd59  1      OPC=nop           
  nop                                                                 #  20    0x9bd5a  1      OPC=nop           
  callq .getwc                                                        #  21    0x9bd5b  5      OPC=callq_label   
  movl %ebx, %ebx                                                     #  22    0x9bd60  2      OPC=movl_r32_r32  
  movl 0x20(%r15,%rbx,1), %esi                                        #  23    0x9bd62  5      OPC=movl_r32_m32  
  movl %eax, %edi                                                     #  24    0x9bd67  2      OPC=movl_r32_r32  
  popq %rbx                                                           #  25    0x9bd69  1      OPC=popq_r64_1    
  jmpq .ungetwc                                                       #  26    0x9bd6a  5      OPC=jmpq_label_1  
  nop                                                                 #  27    0x9bd6f  1      OPC=nop           
  nop                                                                 #  28    0x9bd70  1      OPC=nop           
  nop                                                                 #  29    0x9bd71  1      OPC=nop           
  nop                                                                 #  30    0x9bd72  1      OPC=nop           
  nop                                                                 #  31    0x9bd73  1      OPC=nop           
  nop                                                                 #  32    0x9bd74  1      OPC=nop           
  nop                                                                 #  33    0x9bd75  1      OPC=nop           
  nop                                                                 #  34    0x9bd76  1      OPC=nop           
  nop                                                                 #  35    0x9bd77  1      OPC=nop           
  nop                                                                 #  36    0x9bd78  1      OPC=nop           
  nop                                                                 #  37    0x9bd79  1      OPC=nop           
  nop                                                                 #  38    0x9bd7a  1      OPC=nop           
  nop                                                                 #  39    0x9bd7b  1      OPC=nop           
  nop                                                                 #  40    0x9bd7c  1      OPC=nop           
  nop                                                                 #  41    0x9bd7d  1      OPC=nop           
  nop                                                                 #  42    0x9bd7e  1      OPC=nop           
  nop                                                                 #  43    0x9bd7f  1      OPC=nop           
                                                                                                                 
.size _ZN9__gnu_cxx18stdio_sync_filebufIwSt11char_traitsIwEE9underflowEv, .-_ZN9__gnu_cxx18stdio_sync_filebufIwSt11char_traitsIwEE9underflowEv

