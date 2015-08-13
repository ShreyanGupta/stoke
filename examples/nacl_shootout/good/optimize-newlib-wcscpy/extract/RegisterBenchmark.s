  .text
  .globl RegisterBenchmark
  .type RegisterBenchmark, @function

#! file-offset 0x6a1c0
#! rip-offset  0x2a1c0
#! capacity    256 bytes

# Text                                #  Line  RIP      Bytes  Opcode              
.RegisterBenchmark:                   #        0x2a1c0  0      OPC=<label>         
  subl $0x8, %esp                     #  1     0x2a1c0  3      OPC=subl_r32_imm8   
  addq %r15, %rsp                     #  2     0x2a1c3  3      OPC=addq_r64_r64    
  movl %edi, %edi                     #  3     0x2a1c6  2      OPC=movl_r32_r32    
  movl %esi, %esi                     #  4     0x2a1c8  2      OPC=movl_r32_r32    
  movl %r8d, %r8d                     #  5     0x2a1ca  3      OPC=movl_r32_r32    
  movl %r9d, %r9d                     #  6     0x2a1cd  3      OPC=movl_r32_r32    
  movl 0x10046faa(%rip), %eax         #  7     0x2a1d0  6      OPC=movl_r32_m32    
  cmpl $0x10, %eax                    #  8     0x2a1d6  3      OPC=cmpl_r32_imm8   
  je .L_2a280                         #  9     0x2a1d9  6      OPC=je_label_1      
  nop                                 #  10    0x2a1df  1      OPC=nop             
  leal (%rax,%rax,2), %r11d           #  11    0x2a1e0  4      OPC=leal_r32_m16    
  shll $0x3, %r11d                    #  12    0x2a1e4  4      OPC=shll_r32_imm8   
  movslq %r11d, %r10                  #  13    0x2a1e8  3      OPC=movslq_r64_r32  
  movl %r10d, %r10d                   #  14    0x2a1eb  3      OPC=movl_r32_r32    
  movl %edi, 0x100711a0(%r15,%r10,1)  #  15    0x2a1ee  8      OPC=movl_m32_r32    
  leaq 0x100711a4(%r10), %rdi         #  16    0x2a1f6  7      OPC=leaq_r64_m16    
  nop                                 #  17    0x2a1fd  1      OPC=nop             
  nop                                 #  18    0x2a1fe  1      OPC=nop             
  nop                                 #  19    0x2a1ff  1      OPC=nop             
  movl %edi, %edi                     #  20    0x2a200  2      OPC=movl_r32_r32    
  movl %esi, (%r15,%rdi,1)            #  21    0x2a202  4      OPC=movl_m32_r32    
  leaq 0x100711a8(%r10), %rsi         #  22    0x2a206  7      OPC=leaq_r64_m16    
  movl %esi, %esi                     #  23    0x2a20d  2      OPC=movl_r32_r32    
  movl %r8d, (%r15,%rsi,1)            #  24    0x2a20f  4      OPC=movl_m32_r32    
  addq $0x100711ac, %r10              #  25    0x2a213  7      OPC=addq_r64_imm32  
  nop                                 #  26    0x2a21a  1      OPC=nop             
  nop                                 #  27    0x2a21b  1      OPC=nop             
  nop                                 #  28    0x2a21c  1      OPC=nop             
  nop                                 #  29    0x2a21d  1      OPC=nop             
  nop                                 #  30    0x2a21e  1      OPC=nop             
  nop                                 #  31    0x2a21f  1      OPC=nop             
  movl %r10d, %r10d                   #  32    0x2a220  3      OPC=movl_r32_r32    
  movl %r9d, (%r15,%r10,1)            #  33    0x2a223  4      OPC=movl_m32_r32    
  leal 0x10(%r11), %esi               #  34    0x2a227  4      OPC=leal_r32_m16    
  movslq %esi, %rsi                   #  35    0x2a22b  3      OPC=movslq_r64_r32  
  movl %esi, %esi                     #  36    0x2a22e  2      OPC=movl_r32_r32    
  movl %ecx, 0x100711a0(%r15,%rsi,1)  #  37    0x2a230  8      OPC=movl_m32_r32    
  addq $0x100711a4, %rsi              #  38    0x2a238  7      OPC=addq_r64_imm32  
  nop                                 #  39    0x2a23f  1      OPC=nop             
  movl %esi, %esi                     #  40    0x2a240  2      OPC=movl_r32_r32    
  movl %edx, (%r15,%rsi,1)            #  41    0x2a242  4      OPC=movl_m32_r32    
  addl $0x1, %eax                     #  42    0x2a246  3      OPC=addl_r32_imm8   
  movl %eax, 0x10046f31(%rip)         #  43    0x2a249  6      OPC=movl_m32_r32    
  addl $0x8, %esp                     #  44    0x2a24f  3      OPC=addl_r32_imm8   
  addq %r15, %rsp                     #  45    0x2a252  3      OPC=addq_r64_r64    
  popq %r11                           #  46    0x2a255  2      OPC=popq_r64_1      
  nop                                 #  47    0x2a257  1      OPC=nop             
  nop                                 #  48    0x2a258  1      OPC=nop             
  nop                                 #  49    0x2a259  1      OPC=nop             
  nop                                 #  50    0x2a25a  1      OPC=nop             
  nop                                 #  51    0x2a25b  1      OPC=nop             
  nop                                 #  52    0x2a25c  1      OPC=nop             
  nop                                 #  53    0x2a25d  1      OPC=nop             
  nop                                 #  54    0x2a25e  1      OPC=nop             
  nop                                 #  55    0x2a25f  1      OPC=nop             
  andl $0xffffffe0, %r11d             #  56    0x2a260  7      OPC=andl_r32_imm32  
  nop                                 #  57    0x2a267  1      OPC=nop             
  nop                                 #  58    0x2a268  1      OPC=nop             
  nop                                 #  59    0x2a269  1      OPC=nop             
  nop                                 #  60    0x2a26a  1      OPC=nop             
  addq %r15, %r11                     #  61    0x2a26b  3      OPC=addq_r64_r64    
  jmpq %r11                           #  62    0x2a26e  3      OPC=jmpq_r64        
  nop                                 #  63    0x2a271  1      OPC=nop             
  nop                                 #  64    0x2a272  1      OPC=nop             
  nop                                 #  65    0x2a273  1      OPC=nop             
  nop                                 #  66    0x2a274  1      OPC=nop             
  nop                                 #  67    0x2a275  1      OPC=nop             
  nop                                 #  68    0x2a276  1      OPC=nop             
  nop                                 #  69    0x2a277  1      OPC=nop             
  nop                                 #  70    0x2a278  1      OPC=nop             
  nop                                 #  71    0x2a279  1      OPC=nop             
  nop                                 #  72    0x2a27a  1      OPC=nop             
  nop                                 #  73    0x2a27b  1      OPC=nop             
  nop                                 #  74    0x2a27c  1      OPC=nop             
  nop                                 #  75    0x2a27d  1      OPC=nop             
  nop                                 #  76    0x2a27e  1      OPC=nop             
  nop                                 #  77    0x2a27f  1      OPC=nop             
  nop                                 #  78    0x2a280  1      OPC=nop             
  nop                                 #  79    0x2a281  1      OPC=nop             
  nop                                 #  80    0x2a282  1      OPC=nop             
  nop                                 #  81    0x2a283  1      OPC=nop             
  nop                                 #  82    0x2a284  1      OPC=nop             
  nop                                 #  83    0x2a285  1      OPC=nop             
  nop                                 #  84    0x2a286  1      OPC=nop             
.L_2a280:                             #        0x2a287  0      OPC=<label>         
  movl $0x10020c90, %edi              #  85    0x2a287  5      OPC=movl_r32_imm32  
  xorl %eax, %eax                     #  86    0x2a28c  2      OPC=xorl_r32_r32    
  nop                                 #  87    0x2a28e  1      OPC=nop             
  nop                                 #  88    0x2a28f  1      OPC=nop             
  nop                                 #  89    0x2a290  1      OPC=nop             
  nop                                 #  90    0x2a291  1      OPC=nop             
  nop                                 #  91    0x2a292  1      OPC=nop             
  nop                                 #  92    0x2a293  1      OPC=nop             
  nop                                 #  93    0x2a294  1      OPC=nop             
  nop                                 #  94    0x2a295  1      OPC=nop             
  nop                                 #  95    0x2a296  1      OPC=nop             
  nop                                 #  96    0x2a297  1      OPC=nop             
  nop                                 #  97    0x2a298  1      OPC=nop             
  nop                                 #  98    0x2a299  1      OPC=nop             
  nop                                 #  99    0x2a29a  1      OPC=nop             
  nop                                 #  100   0x2a29b  1      OPC=nop             
  nop                                 #  101   0x2a29c  1      OPC=nop             
  nop                                 #  102   0x2a29d  1      OPC=nop             
  nop                                 #  103   0x2a29e  1      OPC=nop             
  nop                                 #  104   0x2a29f  1      OPC=nop             
  nop                                 #  105   0x2a2a0  1      OPC=nop             
  nop                                 #  106   0x2a2a1  1      OPC=nop             
  callq .ReportStatus                 #  107   0x2a2a2  5      OPC=callq_label     
  movl $0x1, %edi                     #  108   0x2a2a7  5      OPC=movl_r32_imm32  
  nop                                 #  109   0x2a2ac  1      OPC=nop             
  nop                                 #  110   0x2a2ad  1      OPC=nop             
  nop                                 #  111   0x2a2ae  1      OPC=nop             
  nop                                 #  112   0x2a2af  1      OPC=nop             
  nop                                 #  113   0x2a2b0  1      OPC=nop             
  nop                                 #  114   0x2a2b1  1      OPC=nop             
  nop                                 #  115   0x2a2b2  1      OPC=nop             
  nop                                 #  116   0x2a2b3  1      OPC=nop             
  nop                                 #  117   0x2a2b4  1      OPC=nop             
  nop                                 #  118   0x2a2b5  1      OPC=nop             
  nop                                 #  119   0x2a2b6  1      OPC=nop             
  nop                                 #  120   0x2a2b7  1      OPC=nop             
  nop                                 #  121   0x2a2b8  1      OPC=nop             
  nop                                 #  122   0x2a2b9  1      OPC=nop             
  nop                                 #  123   0x2a2ba  1      OPC=nop             
  nop                                 #  124   0x2a2bb  1      OPC=nop             
  nop                                 #  125   0x2a2bc  1      OPC=nop             
  nop                                 #  126   0x2a2bd  1      OPC=nop             
  nop                                 #  127   0x2a2be  1      OPC=nop             
  nop                                 #  128   0x2a2bf  1      OPC=nop             
  nop                                 #  129   0x2a2c0  1      OPC=nop             
  nop                                 #  130   0x2a2c1  1      OPC=nop             
  callq .exit                         #  131   0x2a2c2  5      OPC=callq_label     
                                                                                   
.size RegisterBenchmark, .-RegisterBenchmark

