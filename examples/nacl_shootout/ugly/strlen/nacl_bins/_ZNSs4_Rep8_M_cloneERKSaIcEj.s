  .text
  .globl _ZNSs4_Rep8_M_cloneERKSaIcEj
  .type _ZNSs4_Rep8_M_cloneERKSaIcEj, @function

#! file-offset 0xeb9a0
#! rip-offset  0xab9a0
#! capacity    256 bytes

# Text                                   #  Line  RIP      Bytes  Opcode    
._ZNSs4_Rep8_M_cloneERKSaIcEj:           #        0xab9a0  0      OPC=0     
  movq %rbx, -0x18(%rsp)                 #  1     0xab9a0  5      OPC=1138  
  movl %edi, %ebx                        #  2     0xab9a5  2      OPC=1157  
  movq %r12, -0x10(%rsp)                 #  3     0xab9a7  5      OPC=1138  
  movl %ebx, %ebx                        #  4     0xab9ac  2      OPC=1157  
  movq %r13, -0x8(%rsp)                  #  5     0xab9ae  5      OPC=1138  
  movl %edx, %edi                        #  6     0xab9b3  2      OPC=1157  
  subl $0x18, %esp                       #  7     0xab9b5  3      OPC=2384  
  addq %r15, %rsp                        #  8     0xab9b8  3      OPC=72    
  nop                                    #  9     0xab9bb  1      OPC=1343  
  nop                                    #  10    0xab9bc  1      OPC=1343  
  nop                                    #  11    0xab9bd  1      OPC=1343  
  nop                                    #  12    0xab9be  1      OPC=1343  
  nop                                    #  13    0xab9bf  1      OPC=1343  
  movl %ebx, %ebx                        #  14    0xab9c0  2      OPC=1157  
  addl (%r15,%rbx,1), %edi               #  15    0xab9c2  4      OPC=66    
  movl %esi, %edx                        #  16    0xab9c6  2      OPC=1157  
  movl %ebx, %ebx                        #  17    0xab9c8  2      OPC=1157  
  movl 0x4(%r15,%rbx,1), %esi            #  18    0xab9ca  5      OPC=1156  
  nop                                    #  19    0xab9cf  1      OPC=1343  
  nop                                    #  20    0xab9d0  1      OPC=1343  
  nop                                    #  21    0xab9d1  1      OPC=1343  
  nop                                    #  22    0xab9d2  1      OPC=1343  
  nop                                    #  23    0xab9d3  1      OPC=1343  
  nop                                    #  24    0xab9d4  1      OPC=1343  
  nop                                    #  25    0xab9d5  1      OPC=1343  
  nop                                    #  26    0xab9d6  1      OPC=1343  
  nop                                    #  27    0xab9d7  1      OPC=1343  
  nop                                    #  28    0xab9d8  1      OPC=1343  
  nop                                    #  29    0xab9d9  1      OPC=1343  
  nop                                    #  30    0xab9da  1      OPC=1343  
  callq ._ZNSs4_Rep9_S_createEjjRKSaIcE  #  31    0xab9db  5      OPC=260   
  movl %ebx, %ebx                        #  32    0xab9e0  2      OPC=1157  
  movl (%r15,%rbx,1), %edx               #  33    0xab9e2  4      OPC=1156  
  movl %eax, %r12d                       #  34    0xab9e6  3      OPC=1157  
  testl %edx, %edx                       #  35    0xab9e9  2      OPC=2436  
  jne .L_aba40                           #  36    0xab9eb  6      OPC=963   
  nop                                    #  37    0xab9f1  1      OPC=1343  
  nop                                    #  38    0xab9f2  1      OPC=1343  
  leal 0xc(%r12), %r13d                  #  39    0xab9f3  5      OPC=1066  
  xorl %eax, %eax                        #  40    0xab9f8  2      OPC=3758  
  nop                                    #  41    0xab9fa  1      OPC=1343  
  nop                                    #  42    0xab9fb  1      OPC=1343  
  nop                                    #  43    0xab9fc  1      OPC=1343  
  nop                                    #  44    0xab9fd  1      OPC=1343  
  nop                                    #  45    0xab9fe  1      OPC=1343  
  nop                                    #  46    0xab9ff  1      OPC=1343  
  nop                                    #  47    0xaba00  1      OPC=1343  
  nop                                    #  48    0xaba01  1      OPC=1343  
  nop                                    #  49    0xaba02  1      OPC=1343  
  nop                                    #  50    0xaba03  1      OPC=1343  
  nop                                    #  51    0xaba04  1      OPC=1343  
  nop                                    #  52    0xaba05  1      OPC=1343  
.L_aba00:                                #        0xaba06  0      OPC=0     
  movl %r12d, %r12d                      #  53    0xaba06  3      OPC=1157  
  movl %eax, (%r15,%r12,1)               #  54    0xaba09  4      OPC=1136  
  addl %r13d, %eax                       #  55    0xaba0d  3      OPC=67    
  movl %r12d, %r12d                      #  56    0xaba10  3      OPC=1157  
  movl $0x0, 0x8(%r15,%r12,1)            #  57    0xaba13  9      OPC=1135  
  movl %eax, %eax                        #  58    0xaba1c  2      OPC=1157  
  movb $0x0, (%r15,%rax,1)               #  59    0xaba1e  5      OPC=1140  
  movl %r13d, %eax                       #  60    0xaba23  3      OPC=1157  
  movq (%rsp), %rbx                      #  61    0xaba26  4      OPC=1161  
  movq 0x8(%rsp), %r12                   #  62    0xaba2a  5      OPC=1161  
  movq 0x10(%rsp), %r13                  #  63    0xaba2f  5      OPC=1161  
  addl $0x18, %esp                       #  64    0xaba34  3      OPC=65    
  addq %r15, %rsp                        #  65    0xaba37  3      OPC=72    
  popq %r11                              #  66    0xaba3a  2      OPC=1694  
  andl $0xffffffe0, %r11d                #  67    0xaba3c  7      OPC=131   
  nop                                    #  68    0xaba43  1      OPC=1343  
  nop                                    #  69    0xaba44  1      OPC=1343  
  nop                                    #  70    0xaba45  1      OPC=1343  
  nop                                    #  71    0xaba46  1      OPC=1343  
  addq %r15, %r11                        #  72    0xaba47  3      OPC=72    
  jmpq %r11                              #  73    0xaba4a  3      OPC=928   
.L_aba40:                                #        0xaba4d  0      OPC=0     
  cmpl $0x1, %edx                        #  74    0xaba4d  3      OPC=470   
  leal 0xc(%rbx), %esi                   #  75    0xaba50  3      OPC=1066  
  leal 0xc(%r12), %r13d                  #  76    0xaba53  5      OPC=1066  
  je .L_aba80                            #  77    0xaba58  6      OPC=893   
  nop                                    #  78    0xaba5e  1      OPC=1343  
  nop                                    #  79    0xaba5f  1      OPC=1343  
  movl %r13d, %edi                       #  80    0xaba60  3      OPC=1157  
  nop                                    #  81    0xaba63  1      OPC=1343  
  nop                                    #  82    0xaba64  1      OPC=1343  
  nop                                    #  83    0xaba65  1      OPC=1343  
  nop                                    #  84    0xaba66  1      OPC=1343  
  nop                                    #  85    0xaba67  1      OPC=1343  
  nop                                    #  86    0xaba68  1      OPC=1343  
  nop                                    #  87    0xaba69  1      OPC=1343  
  nop                                    #  88    0xaba6a  1      OPC=1343  
  nop                                    #  89    0xaba6b  1      OPC=1343  
  nop                                    #  90    0xaba6c  1      OPC=1343  
  nop                                    #  91    0xaba6d  1      OPC=1343  
  callq .memcpy                          #  92    0xaba6e  5      OPC=260   
  movl %ebx, %ebx                        #  93    0xaba73  2      OPC=1157  
  movl (%r15,%rbx,1), %eax               #  94    0xaba75  4      OPC=1156  
  jmpq .L_aba00                          #  95    0xaba79  5      OPC=930   
  nop                                    #  96    0xaba7e  1      OPC=1343  
  nop                                    #  97    0xaba7f  1      OPC=1343  
  nop                                    #  98    0xaba80  1      OPC=1343  
  nop                                    #  99    0xaba81  1      OPC=1343  
  nop                                    #  100   0xaba82  1      OPC=1343  
  nop                                    #  101   0xaba83  1      OPC=1343  
  nop                                    #  102   0xaba84  1      OPC=1343  
  nop                                    #  103   0xaba85  1      OPC=1343  
  nop                                    #  104   0xaba86  1      OPC=1343  
  nop                                    #  105   0xaba87  1      OPC=1343  
  nop                                    #  106   0xaba88  1      OPC=1343  
  nop                                    #  107   0xaba89  1      OPC=1343  
  nop                                    #  108   0xaba8a  1      OPC=1343  
  nop                                    #  109   0xaba8b  1      OPC=1343  
  nop                                    #  110   0xaba8c  1      OPC=1343  
  nop                                    #  111   0xaba8d  1      OPC=1343  
  nop                                    #  112   0xaba8e  1      OPC=1343  
  nop                                    #  113   0xaba8f  1      OPC=1343  
  nop                                    #  114   0xaba90  1      OPC=1343  
  nop                                    #  115   0xaba91  1      OPC=1343  
  nop                                    #  116   0xaba92  1      OPC=1343  
  nop                                    #  117   0xaba93  1      OPC=1343  
  nop                                    #  118   0xaba94  1      OPC=1343  
  nop                                    #  119   0xaba95  1      OPC=1343  
  nop                                    #  120   0xaba96  1      OPC=1343  
  nop                                    #  121   0xaba97  1      OPC=1343  
.L_aba80:                                #        0xaba98  0      OPC=0     
  movl %esi, %esi                        #  122   0xaba98  2      OPC=1157  
  movzbl (%r15,%rsi,1), %eax             #  123   0xaba9a  5      OPC=1302  
  movl %r13d, %r13d                      #  124   0xaba9f  3      OPC=1157  
  movb %al, (%r15,%r13,1)                #  125   0xabaa2  4      OPC=1141  
  movl %ebx, %ebx                        #  126   0xabaa6  2      OPC=1157  
  movl (%r15,%rbx,1), %eax               #  127   0xabaa8  4      OPC=1156  
  jmpq .L_aba00                          #  128   0xabaac  5      OPC=930   
  nop                                    #  129   0xabab1  1      OPC=1343  
  nop                                    #  130   0xabab2  1      OPC=1343  
  nop                                    #  131   0xabab3  1      OPC=1343  
  nop                                    #  132   0xabab4  1      OPC=1343  
  nop                                    #  133   0xabab5  1      OPC=1343  
  nop                                    #  134   0xabab6  1      OPC=1343  
  nop                                    #  135   0xabab7  1      OPC=1343  
                                                                            
.size _ZNSs4_Rep8_M_cloneERKSaIcEj, .-_ZNSs4_Rep8_M_cloneERKSaIcEj
