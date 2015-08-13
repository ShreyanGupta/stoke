  .text
  .globl frexp
  .type frexp, @function

#! file-offset 0x18bba0
#! rip-offset  0x14bba0
#! capacity    288 bytes

# Text                             #  Line  RIP       Bytes  Opcode                
.frexp:                            #        0x14bba0  0      OPC=<label>           
  movsd %xmm0, -0x8(%rsp)          #  1     0x14bba0  6      OPC=movsd_m64_xmm     
  movq -0x8(%rsp), %rax            #  2     0x14bba6  5      OPC=movq_r64_m64      
  movl %edi, %edi                  #  3     0x14bbab  2      OPC=movl_r32_r32      
  movl %edi, %edi                  #  4     0x14bbad  2      OPC=movl_r32_r32      
  movl $0x0, (%r15,%rdi,1)         #  5     0x14bbaf  8      OPC=movl_m32_imm32    
  movq %rax, %rdx                  #  6     0x14bbb7  3      OPC=movq_r64_r64      
  movl %eax, %ecx                  #  7     0x14bbba  2      OPC=movl_r32_r32      
  shrq $0x20, %rdx                 #  8     0x14bbbc  4      OPC=shrq_r64_imm8     
  movl %edx, %eax                  #  9     0x14bbc0  2      OPC=movl_r32_r32      
  andl $0x7fffffff, %eax           #  10    0x14bbc2  5      OPC=andl_eax_imm32    
  cmpl $0x7fefffff, %eax           #  11    0x14bbc7  5      OPC=cmpl_eax_imm32    
  jg .L_14bc40                     #  12    0x14bbcc  2      OPC=jg_label          
  testl %eax, %eax                 #  13    0x14bbce  2      OPC=testl_r32_r32     
  je .L_14bc60                     #  14    0x14bbd0  6      OPC=je_label_1        
  cmpl $0xfffff, %eax              #  15    0x14bbd6  5      OPC=cmpl_eax_imm32    
  movl $0xfffffc02, %ecx           #  16    0x14bbdb  6      OPC=movl_r32_imm32_1  
  jle .L_14bc80                    #  17    0x14bbe1  6      OPC=jle_label_1       
  nop                              #  18    0x14bbe7  1      OPC=nop               
  nop                              #  19    0x14bbe8  1      OPC=nop               
  nop                              #  20    0x14bbe9  1      OPC=nop               
  nop                              #  21    0x14bbea  1      OPC=nop               
  nop                              #  22    0x14bbeb  1      OPC=nop               
  nop                              #  23    0x14bbec  1      OPC=nop               
  nop                              #  24    0x14bbed  1      OPC=nop               
  nop                              #  25    0x14bbee  1      OPC=nop               
  nop                              #  26    0x14bbef  1      OPC=nop               
  nop                              #  27    0x14bbf0  1      OPC=nop               
  nop                              #  28    0x14bbf1  1      OPC=nop               
  nop                              #  29    0x14bbf2  1      OPC=nop               
  nop                              #  30    0x14bbf3  1      OPC=nop               
  nop                              #  31    0x14bbf4  1      OPC=nop               
  nop                              #  32    0x14bbf5  1      OPC=nop               
  nop                              #  33    0x14bbf6  1      OPC=nop               
  nop                              #  34    0x14bbf7  1      OPC=nop               
  nop                              #  35    0x14bbf8  1      OPC=nop               
  nop                              #  36    0x14bbf9  1      OPC=nop               
  nop                              #  37    0x14bbfa  1      OPC=nop               
  nop                              #  38    0x14bbfb  1      OPC=nop               
  nop                              #  39    0x14bbfc  1      OPC=nop               
  nop                              #  40    0x14bbfd  1      OPC=nop               
  nop                              #  41    0x14bbfe  1      OPC=nop               
  nop                              #  42    0x14bbff  1      OPC=nop               
  nop                              #  43    0x14bc00  1      OPC=nop               
.L_14bc00:                         #        0x14bc01  0      OPC=<label>           
  sarl $0x14, %eax                 #  44    0x14bc01  3      OPC=sarl_r32_imm8     
  movsd %xmm0, -0x8(%rsp)          #  45    0x14bc04  6      OPC=movsd_m64_xmm     
  andl $0x800fffff, %edx           #  46    0x14bc0a  6      OPC=andl_r32_imm32    
  leal (%rax,%rcx,1), %ecx         #  47    0x14bc10  3      OPC=leal_r32_m16      
  movq -0x8(%rsp), %rax            #  48    0x14bc13  5      OPC=movq_r64_m64      
  orq $0x3fe00000, %rdx            #  49    0x14bc18  7      OPC=orq_r64_imm32     
  xchgw %ax, %ax                   #  50    0x14bc1f  2      OPC=xchgw_ax_r16      
  shlq $0x20, %rdx                 #  51    0x14bc21  4      OPC=shlq_r64_imm8     
  movl %edi, %edi                  #  52    0x14bc25  2      OPC=movl_r32_r32      
  movl %ecx, (%r15,%rdi,1)         #  53    0x14bc27  4      OPC=movl_m32_r32      
  andl $0xffffffff, %eax           #  54    0x14bc2b  6      OPC=andl_r32_imm32    
  nop                              #  55    0x14bc31  1      OPC=nop               
  nop                              #  56    0x14bc32  1      OPC=nop               
  nop                              #  57    0x14bc33  1      OPC=nop               
  orq %rdx, %rax                   #  58    0x14bc34  3      OPC=orq_r64_r64       
  movq %rax, -0x8(%rsp)            #  59    0x14bc37  5      OPC=movq_m64_r64      
  movsd -0x8(%rsp), %xmm0          #  60    0x14bc3c  6      OPC=movsd_xmm_m64     
  nop                              #  61    0x14bc42  1      OPC=nop               
  nop                              #  62    0x14bc43  1      OPC=nop               
  nop                              #  63    0x14bc44  1      OPC=nop               
  nop                              #  64    0x14bc45  1      OPC=nop               
  nop                              #  65    0x14bc46  1      OPC=nop               
.L_14bc40:                         #        0x14bc47  0      OPC=<label>           
  popq %r11                        #  66    0x14bc47  2      OPC=popq_r64_1        
  andl $0xffffffe0, %r11d          #  67    0x14bc49  7      OPC=andl_r32_imm32    
  nop                              #  68    0x14bc50  1      OPC=nop               
  nop                              #  69    0x14bc51  1      OPC=nop               
  nop                              #  70    0x14bc52  1      OPC=nop               
  nop                              #  71    0x14bc53  1      OPC=nop               
  addq %r15, %r11                  #  72    0x14bc54  3      OPC=addq_r64_r64      
  jmpq %r11                        #  73    0x14bc57  3      OPC=jmpq_r64          
  nop                              #  74    0x14bc5a  1      OPC=nop               
  nop                              #  75    0x14bc5b  1      OPC=nop               
  nop                              #  76    0x14bc5c  1      OPC=nop               
  nop                              #  77    0x14bc5d  1      OPC=nop               
  nop                              #  78    0x14bc5e  1      OPC=nop               
  nop                              #  79    0x14bc5f  1      OPC=nop               
  nop                              #  80    0x14bc60  1      OPC=nop               
  nop                              #  81    0x14bc61  1      OPC=nop               
  nop                              #  82    0x14bc62  1      OPC=nop               
  nop                              #  83    0x14bc63  1      OPC=nop               
  nop                              #  84    0x14bc64  1      OPC=nop               
  nop                              #  85    0x14bc65  1      OPC=nop               
  nop                              #  86    0x14bc66  1      OPC=nop               
  nop                              #  87    0x14bc67  1      OPC=nop               
  nop                              #  88    0x14bc68  1      OPC=nop               
  nop                              #  89    0x14bc69  1      OPC=nop               
  nop                              #  90    0x14bc6a  1      OPC=nop               
  nop                              #  91    0x14bc6b  1      OPC=nop               
  nop                              #  92    0x14bc6c  1      OPC=nop               
  nop                              #  93    0x14bc6d  1      OPC=nop               
.L_14bc60:                         #        0x14bc6e  0      OPC=<label>           
  testl %ecx, %ecx                 #  94    0x14bc6e  2      OPC=testl_r32_r32     
  je .L_14bc40                     #  95    0x14bc70  2      OPC=je_label          
  nop                              #  96    0x14bc72  1      OPC=nop               
  nop                              #  97    0x14bc73  1      OPC=nop               
  nop                              #  98    0x14bc74  1      OPC=nop               
  nop                              #  99    0x14bc75  1      OPC=nop               
  nop                              #  100   0x14bc76  1      OPC=nop               
  nop                              #  101   0x14bc77  1      OPC=nop               
  nop                              #  102   0x14bc78  1      OPC=nop               
  nop                              #  103   0x14bc79  1      OPC=nop               
  nop                              #  104   0x14bc7a  1      OPC=nop               
  nop                              #  105   0x14bc7b  1      OPC=nop               
  nop                              #  106   0x14bc7c  1      OPC=nop               
  nop                              #  107   0x14bc7d  1      OPC=nop               
  nop                              #  108   0x14bc7e  1      OPC=nop               
  nop                              #  109   0x14bc7f  1      OPC=nop               
  nop                              #  110   0x14bc80  1      OPC=nop               
  nop                              #  111   0x14bc81  1      OPC=nop               
  nop                              #  112   0x14bc82  1      OPC=nop               
  nop                              #  113   0x14bc83  1      OPC=nop               
  nop                              #  114   0x14bc84  1      OPC=nop               
  nop                              #  115   0x14bc85  1      OPC=nop               
  nop                              #  116   0x14bc86  1      OPC=nop               
  nop                              #  117   0x14bc87  1      OPC=nop               
  nop                              #  118   0x14bc88  1      OPC=nop               
  nop                              #  119   0x14bc89  1      OPC=nop               
  nop                              #  120   0x14bc8a  1      OPC=nop               
  nop                              #  121   0x14bc8b  1      OPC=nop               
  nop                              #  122   0x14bc8c  1      OPC=nop               
  nop                              #  123   0x14bc8d  1      OPC=nop               
.L_14bc80:                         #        0x14bc8e  0      OPC=<label>           
  mulsd 0xfef4578(%rip), %xmm0     #  124   0x14bc8e  8      OPC=mulsd_xmm_m64     
  movl %edi, %edi                  #  125   0x14bc96  2      OPC=movl_r32_r32      
  movl $0xffffffca, (%r15,%rdi,1)  #  126   0x14bc98  8      OPC=movl_m32_imm32    
  movl $0xfffffbcc, %ecx           #  127   0x14bca0  6      OPC=movl_r32_imm32_1  
  movsd %xmm0, -0x8(%rsp)          #  128   0x14bca6  6      OPC=movsd_m64_xmm     
  nop                              #  129   0x14bcac  1      OPC=nop               
  nop                              #  130   0x14bcad  1      OPC=nop               
  nop                              #  131   0x14bcae  1      OPC=nop               
  movq -0x8(%rsp), %rdx            #  132   0x14bcaf  5      OPC=movq_r64_m64      
  shrq $0x20, %rdx                 #  133   0x14bcb4  4      OPC=shrq_r64_imm8     
  movl %edx, %eax                  #  134   0x14bcb8  2      OPC=movl_r32_r32      
  andl $0x7fffffff, %eax           #  135   0x14bcba  5      OPC=andl_eax_imm32    
  jmpq .L_14bc00                   #  136   0x14bcbf  5      OPC=jmpq_label_1      
  nop                              #  137   0x14bcc4  1      OPC=nop               
  nop                              #  138   0x14bcc5  1      OPC=nop               
  nop                              #  139   0x14bcc6  1      OPC=nop               
  nop                              #  140   0x14bcc7  1      OPC=nop               
  nop                              #  141   0x14bcc8  1      OPC=nop               
  nop                              #  142   0x14bcc9  1      OPC=nop               
  nop                              #  143   0x14bcca  1      OPC=nop               
  nop                              #  144   0x14bccb  1      OPC=nop               
  nop                              #  145   0x14bccc  1      OPC=nop               
  nop                              #  146   0x14bccd  1      OPC=nop               
  nop                              #  147   0x14bcce  1      OPC=nop               
                                                                                   
.size frexp, .-frexp

