  .text
  .globl print_freqs
  .type print_freqs, @function

#! file-offset 0x64a20
#! rip-offset  0x24a20
#! capacity    704 bytes

# Text                                  #  Line  RIP      Bytes  Opcode                 
.print_freqs:                           #        0x24a20  0      OPC=<label>            
  pushq %rbp                            #  1     0x24a20  1      OPC=pushq_r64_1        
  movq %rsp, %rbp                       #  2     0x24a21  3      OPC=movq_r64_r64       
  pushq %r14                            #  3     0x24a24  2      OPC=pushq_r64_1        
  pushq %r13                            #  4     0x24a26  2      OPC=pushq_r64_1        
  pushq %r12                            #  5     0x24a28  2      OPC=pushq_r64_1        
  pushq %rbx                            #  6     0x24a2a  1      OPC=pushq_r64_1        
  subl $0x40, %esp                      #  7     0x24a2b  3      OPC=subl_r32_imm8      
  addq %r15, %rsp                       #  8     0x24a2e  3      OPC=addq_r64_r64       
  movl %edi, %eax                       #  9     0x24a31  2      OPC=movl_r32_r32       
  movl %eax, %eax                       #  10    0x24a33  2      OPC=movl_r32_r32       
  movl (%r15,%rax,1), %r13d             #  11    0x24a35  4      OPC=movl_r32_m32       
  movl %eax, %eax                       #  12    0x24a39  2      OPC=movl_r32_r32       
  movl 0x4(%r15,%rax,1), %r12d          #  13    0x24a3b  5      OPC=movl_r32_m32       
  movl %eax, %eax                       #  14    0x24a40  2      OPC=movl_r32_r32       
  movl 0x8(%r15,%rax,1), %edx           #  15    0x24a42  5      OPC=movl_r32_m32       
  movl %edx, -0x28(%rbp)                #  16    0x24a47  3      OPC=movl_m32_r32       
  movl %eax, %eax                       #  17    0x24a4a  2      OPC=movl_r32_r32       
  movl 0xc(%r15,%rax,1), %ecx           #  18    0x24a4c  5      OPC=movl_r32_m32       
  movq %rcx, -0x40(%rbp)                #  19    0x24a51  4      OPC=movq_m64_r64       
  movl %eax, %eax                       #  20    0x24a55  2      OPC=movl_r32_r32       
  movl 0x10(%r15,%rax,1), %eax          #  21    0x24a57  5      OPC=movl_r32_m32       
  movl %eax, -0x34(%rbp)                #  22    0x24a5c  3      OPC=movl_m32_r32       
  nop                                   #  23    0x24a5f  1      OPC=nop                
  movl $0x20, %edi                      #  24    0x24a60  5      OPC=movl_r32_imm32     
  nop                                   #  25    0x24a65  1      OPC=nop                
  nop                                   #  26    0x24a66  1      OPC=nop                
  nop                                   #  27    0x24a67  1      OPC=nop                
  nop                                   #  28    0x24a68  1      OPC=nop                
  nop                                   #  29    0x24a69  1      OPC=nop                
  nop                                   #  30    0x24a6a  1      OPC=nop                
  nop                                   #  31    0x24a6b  1      OPC=nop                
  nop                                   #  32    0x24a6c  1      OPC=nop                
  nop                                   #  33    0x24a6d  1      OPC=nop                
  nop                                   #  34    0x24a6e  1      OPC=nop                
  nop                                   #  35    0x24a6f  1      OPC=nop                
  nop                                   #  36    0x24a70  1      OPC=nop                
  nop                                   #  37    0x24a71  1      OPC=nop                
  nop                                   #  38    0x24a72  1      OPC=nop                
  nop                                   #  39    0x24a73  1      OPC=nop                
  nop                                   #  40    0x24a74  1      OPC=nop                
  nop                                   #  41    0x24a75  1      OPC=nop                
  nop                                   #  42    0x24a76  1      OPC=nop                
  nop                                   #  43    0x24a77  1      OPC=nop                
  nop                                   #  44    0x24a78  1      OPC=nop                
  nop                                   #  45    0x24a79  1      OPC=nop                
  nop                                   #  46    0x24a7a  1      OPC=nop                
  callq .ht_create                      #  47    0x24a7b  5      OPC=callq_label        
  movl %eax, %eax                       #  48    0x24a80  2      OPC=movl_r32_r32       
  movq %rax, -0x58(%rbp)                #  49    0x24a82  4      OPC=movq_m64_r64       
  movl -0x28(%rbp), %eax                #  50    0x24a86  3      OPC=movl_r32_m32       
  addl $0x1, %eax                       #  51    0x24a89  3      OPC=addl_r32_imm8      
  addq $0x1e, %rax                      #  52    0x24a8c  4      OPC=addq_r64_imm8      
  shrq $0x4, %rax                       #  53    0x24a90  4      OPC=shrq_r64_imm8      
  shlq $0x4, %rax                       #  54    0x24a94  4      OPC=shlq_r64_imm8      
  subl %eax, %esp                       #  55    0x24a98  2      OPC=subl_r32_r32       
  addq %r15, %rsp                       #  56    0x24a9a  3      OPC=addq_r64_r64       
  nop                                   #  57    0x24a9d  1      OPC=nop                
  nop                                   #  58    0x24a9e  1      OPC=nop                
  nop                                   #  59    0x24a9f  1      OPC=nop                
  leal 0xf(%rsp), %ebx                  #  60    0x24aa0  4      OPC=leal_r32_m16       
  andl $0xfffffff0, %ebx                #  61    0x24aa4  6      OPC=andl_r32_imm32     
  nop                                   #  62    0x24aaa  1      OPC=nop                
  nop                                   #  63    0x24aab  1      OPC=nop                
  nop                                   #  64    0x24aac  1      OPC=nop                
  movl -0x58(%rbp), %ecx                #  65    0x24aad  3      OPC=movl_r32_m32       
  movl -0x28(%rbp), %edx                #  66    0x24ab0  3      OPC=movl_r32_m32       
  movl %r12d, %esi                      #  67    0x24ab3  3      OPC=movl_r32_r32       
  movl %r13d, %edi                      #  68    0x24ab6  3      OPC=movl_r32_r32       
  nop                                   #  69    0x24ab9  1      OPC=nop                
  nop                                   #  70    0x24aba  1      OPC=nop                
  nop                                   #  71    0x24abb  1      OPC=nop                
  nop                                   #  72    0x24abc  1      OPC=nop                
  nop                                   #  73    0x24abd  1      OPC=nop                
  nop                                   #  74    0x24abe  1      OPC=nop                
  nop                                   #  75    0x24abf  1      OPC=nop                
  nop                                   #  76    0x24ac0  1      OPC=nop                
  callq .generate_seqences              #  77    0x24ac1  5      OPC=callq_label        
  movl -0x58(%rbp), %edi                #  78    0x24ac6  3      OPC=movl_r32_m32       
  nop                                   #  79    0x24ac9  1      OPC=nop                
  nop                                   #  80    0x24aca  1      OPC=nop                
  nop                                   #  81    0x24acb  1      OPC=nop                
  nop                                   #  82    0x24acc  1      OPC=nop                
  nop                                   #  83    0x24acd  1      OPC=nop                
  nop                                   #  84    0x24ace  1      OPC=nop                
  nop                                   #  85    0x24acf  1      OPC=nop                
  nop                                   #  86    0x24ad0  1      OPC=nop                
  nop                                   #  87    0x24ad1  1      OPC=nop                
  nop                                   #  88    0x24ad2  1      OPC=nop                
  nop                                   #  89    0x24ad3  1      OPC=nop                
  nop                                   #  90    0x24ad4  1      OPC=nop                
  nop                                   #  91    0x24ad5  1      OPC=nop                
  nop                                   #  92    0x24ad6  1      OPC=nop                
  nop                                   #  93    0x24ad7  1      OPC=nop                
  nop                                   #  94    0x24ad8  1      OPC=nop                
  nop                                   #  95    0x24ad9  1      OPC=nop                
  nop                                   #  96    0x24ada  1      OPC=nop                
  nop                                   #  97    0x24adb  1      OPC=nop                
  nop                                   #  98    0x24adc  1      OPC=nop                
  nop                                   #  99    0x24add  1      OPC=nop                
  nop                                   #  100   0x24ade  1      OPC=nop                
  nop                                   #  101   0x24adf  1      OPC=nop                
  nop                                   #  102   0x24ae0  1      OPC=nop                
  callq .ht_values_as_vector            #  103   0x24ae1  5      OPC=callq_label        
  movl %eax, %eax                       #  104   0x24ae6  2      OPC=movl_r32_r32       
  movq %rax, -0x50(%rbp)                #  105   0x24ae8  4      OPC=movq_m64_r64       
  movq -0x58(%rbp), %rax                #  106   0x24aec  4      OPC=movq_r64_m64       
  movl %eax, %eax                       #  107   0x24af0  2      OPC=movl_r32_r32       
  movl 0x10(%r15,%rax,1), %eax          #  108   0x24af2  5      OPC=movl_r32_m32       
  movl %eax, -0x24(%rbp)                #  109   0x24af7  3      OPC=movl_m32_r32       
  movl $0x235a0, %ecx                   #  110   0x24afa  5      OPC=movl_r32_imm32     
  movl $0x10, %edx                      #  111   0x24aff  5      OPC=movl_r32_imm32     
  movl %eax, %esi                       #  112   0x24b04  2      OPC=movl_r32_r32       
  movl -0x50(%rbp), %edi                #  113   0x24b06  3      OPC=movl_r32_m32       
  nop                                   #  114   0x24b09  1      OPC=nop                
  nop                                   #  115   0x24b0a  1      OPC=nop                
  nop                                   #  116   0x24b0b  1      OPC=nop                
  nop                                   #  117   0x24b0c  1      OPC=nop                
  nop                                   #  118   0x24b0d  1      OPC=nop                
  nop                                   #  119   0x24b0e  1      OPC=nop                
  nop                                   #  120   0x24b0f  1      OPC=nop                
  nop                                   #  121   0x24b10  1      OPC=nop                
  nop                                   #  122   0x24b11  1      OPC=nop                
  nop                                   #  123   0x24b12  1      OPC=nop                
  nop                                   #  124   0x24b13  1      OPC=nop                
  nop                                   #  125   0x24b14  1      OPC=nop                
  nop                                   #  126   0x24b15  1      OPC=nop                
  nop                                   #  127   0x24b16  1      OPC=nop                
  nop                                   #  128   0x24b17  1      OPC=nop                
  nop                                   #  129   0x24b18  1      OPC=nop                
  nop                                   #  130   0x24b19  1      OPC=nop                
  nop                                   #  131   0x24b1a  1      OPC=nop                
  nop                                   #  132   0x24b1b  1      OPC=nop                
  nop                                   #  133   0x24b1c  1      OPC=nop                
  nop                                   #  134   0x24b1d  1      OPC=nop                
  nop                                   #  135   0x24b1e  1      OPC=nop                
  nop                                   #  136   0x24b1f  1      OPC=nop                
  nop                                   #  137   0x24b20  1      OPC=nop                
  callq .qsort                          #  138   0x24b21  5      OPC=callq_label        
  movl -0x24(%rbp), %edi                #  139   0x24b26  3      OPC=movl_r32_m32       
  testl %edi, %edi                      #  140   0x24b29  2      OPC=testl_r32_r32      
  jle .L_24c60                          #  141   0x24b2b  6      OPC=jle_label_1        
  movl -0x50(%rbp), %r13d               #  142   0x24b31  4      OPC=movl_r32_m32       
  movl %r13d, %ecx                      #  143   0x24b35  3      OPC=movl_r32_r32       
  xorl %eax, %eax                       #  144   0x24b38  2      OPC=xorl_r32_r32       
  xorl %edx, %edx                       #  145   0x24b3a  2      OPC=xorl_r32_r32       
  movl -0x24(%rbp), %edi                #  146   0x24b3c  3      OPC=movl_r32_m32       
  nop                                   #  147   0x24b3f  1      OPC=nop                
  nop                                   #  148   0x24b40  1      OPC=nop                
  nop                                   #  149   0x24b41  1      OPC=nop                
  nop                                   #  150   0x24b42  1      OPC=nop                
  nop                                   #  151   0x24b43  1      OPC=nop                
  nop                                   #  152   0x24b44  1      OPC=nop                
  nop                                   #  153   0x24b45  1      OPC=nop                
.L_24b40:                               #        0x24b46  0      OPC=<label>            
  movl %ecx, %esi                       #  154   0x24b46  2      OPC=movl_r32_r32       
  movl %esi, %esi                       #  155   0x24b48  2      OPC=movl_r32_r32       
  addl 0x8(%r15,%rsi,1), %edx           #  156   0x24b4a  5      OPC=addl_r32_m32       
  addl $0x1, %eax                       #  157   0x24b4f  3      OPC=addl_r32_imm8      
  addl $0x10, %ecx                      #  158   0x24b52  3      OPC=addl_r32_imm8      
  cmpl %edi, %eax                       #  159   0x24b55  2      OPC=cmpl_r32_r32       
  jne .L_24b40                          #  160   0x24b57  2      OPC=jne_label          
  movl -0x28(%rbp), %eax                #  161   0x24b59  3      OPC=movl_r32_m32       
  addl %ebx, %eax                       #  162   0x24b5c  2      OPC=addl_r32_r32       
  movl %eax, %ecx                       #  163   0x24b5e  2      OPC=movl_r32_r32       
  movq %rcx, -0x30(%rbp)                #  164   0x24b60  4      OPC=movq_m64_r64       
  xchgw %ax, %ax                        #  165   0x24b64  2      OPC=xchgw_ax_r16       
  xorl %r14d, %r14d                     #  166   0x24b66  3      OPC=xorl_r32_r32       
  xorl %r12d, %r12d                     #  167   0x24b69  3      OPC=xorl_r32_r32       
  cvtsi2ssl %edx, %xmm0                 #  168   0x24b6c  4      OPC=cvtsi2ssl_xmm_r32  
  movss %xmm0, -0x44(%rbp)              #  169   0x24b70  5      OPC=movss_m32_xmm      
  movl %eax, -0x48(%rbp)                #  170   0x24b75  3      OPC=movl_m32_r32       
  nop                                   #  171   0x24b78  1      OPC=nop                
  nop                                   #  172   0x24b79  1      OPC=nop                
  nop                                   #  173   0x24b7a  1      OPC=nop                
  nop                                   #  174   0x24b7b  1      OPC=nop                
  nop                                   #  175   0x24b7c  1      OPC=nop                
  nop                                   #  176   0x24b7d  1      OPC=nop                
  nop                                   #  177   0x24b7e  1      OPC=nop                
  nop                                   #  178   0x24b7f  1      OPC=nop                
  nop                                   #  179   0x24b80  1      OPC=nop                
  nop                                   #  180   0x24b81  1      OPC=nop                
  nop                                   #  181   0x24b82  1      OPC=nop                
  nop                                   #  182   0x24b83  1      OPC=nop                
  nop                                   #  183   0x24b84  1      OPC=nop                
  nop                                   #  184   0x24b85  1      OPC=nop                
.L_24b80:                               #        0x24b86  0      OPC=<label>            
  movl %r13d, %edi                      #  185   0x24b86  3      OPC=movl_r32_r32       
  movl %edi, %edi                       #  186   0x24b89  2      OPC=movl_r32_r32       
  movq (%r15,%rdi,1), %rax              #  187   0x24b8b  4      OPC=movq_r64_m64       
  movl -0x28(%rbp), %esi                #  188   0x24b8f  3      OPC=movl_r32_m32       
  testl %esi, %esi                      #  189   0x24b92  2      OPC=testl_r32_r32      
  jle .L_24be0                          #  190   0x24b94  2      OPC=jle_label          
  movl -0x48(%rbp), %edx                #  191   0x24b96  3      OPC=movl_r32_m32       
  nop                                   #  192   0x24b99  1      OPC=nop                
  nop                                   #  193   0x24b9a  1      OPC=nop                
  nop                                   #  194   0x24b9b  1      OPC=nop                
  nop                                   #  195   0x24b9c  1      OPC=nop                
  nop                                   #  196   0x24b9d  1      OPC=nop                
  nop                                   #  197   0x24b9e  1      OPC=nop                
  nop                                   #  198   0x24b9f  1      OPC=nop                
  nop                                   #  199   0x24ba0  1      OPC=nop                
  nop                                   #  200   0x24ba1  1      OPC=nop                
  nop                                   #  201   0x24ba2  1      OPC=nop                
  nop                                   #  202   0x24ba3  1      OPC=nop                
  nop                                   #  203   0x24ba4  1      OPC=nop                
  nop                                   #  204   0x24ba5  1      OPC=nop                
.L_24ba0:                               #        0x24ba6  0      OPC=<label>            
  subl $0x1, %edx                       #  205   0x24ba6  3      OPC=subl_r32_imm8      
  movl %edx, %ecx                       #  206   0x24ba9  2      OPC=movl_r32_r32       
  movq %rax, %rsi                       #  207   0x24bab  3      OPC=movq_r64_r64       
  andl $0x3, %esi                       #  208   0x24bae  3      OPC=andl_r32_imm8      
  movl %esi, %esi                       #  209   0x24bb1  2      OPC=movl_r32_r32       
  movzbl 0x10020548(%r15,%rsi,1), %esi  #  210   0x24bb3  9      OPC=movzbl_r32_m8      
  movl %ecx, %ecx                       #  211   0x24bbc  2      OPC=movl_r32_r32       
  movb %sil, (%r15,%rcx,1)              #  212   0x24bbe  4      OPC=movb_m8_r8         
  shrq $0x2, %rax                       #  213   0x24bc2  4      OPC=shrq_r64_imm8      
  cmpl %ebx, %edx                       #  214   0x24bc6  2      OPC=cmpl_r32_r32       
  jne .L_24ba0                          #  215   0x24bc8  2      OPC=jne_label          
  nop                                   #  216   0x24bca  1      OPC=nop                
  nop                                   #  217   0x24bcb  1      OPC=nop                
  nop                                   #  218   0x24bcc  1      OPC=nop                
  nop                                   #  219   0x24bcd  1      OPC=nop                
  nop                                   #  220   0x24bce  1      OPC=nop                
  nop                                   #  221   0x24bcf  1      OPC=nop                
  nop                                   #  222   0x24bd0  1      OPC=nop                
  nop                                   #  223   0x24bd1  1      OPC=nop                
  nop                                   #  224   0x24bd2  1      OPC=nop                
  nop                                   #  225   0x24bd3  1      OPC=nop                
  nop                                   #  226   0x24bd4  1      OPC=nop                
  nop                                   #  227   0x24bd5  1      OPC=nop                
  nop                                   #  228   0x24bd6  1      OPC=nop                
  nop                                   #  229   0x24bd7  1      OPC=nop                
  nop                                   #  230   0x24bd8  1      OPC=nop                
  nop                                   #  231   0x24bd9  1      OPC=nop                
  nop                                   #  232   0x24bda  1      OPC=nop                
  nop                                   #  233   0x24bdb  1      OPC=nop                
  nop                                   #  234   0x24bdc  1      OPC=nop                
  nop                                   #  235   0x24bdd  1      OPC=nop                
  nop                                   #  236   0x24bde  1      OPC=nop                
  nop                                   #  237   0x24bdf  1      OPC=nop                
  nop                                   #  238   0x24be0  1      OPC=nop                
  nop                                   #  239   0x24be1  1      OPC=nop                
  nop                                   #  240   0x24be2  1      OPC=nop                
  nop                                   #  241   0x24be3  1      OPC=nop                
  nop                                   #  242   0x24be4  1      OPC=nop                
  nop                                   #  243   0x24be5  1      OPC=nop                
.L_24be0:                               #        0x24be6  0      OPC=<label>            
  movq -0x30(%rbp), %rax                #  244   0x24be6  4      OPC=movq_r64_m64       
  movl %eax, %eax                       #  245   0x24bea  2      OPC=movl_r32_r32       
  movb $0x0, (%r15,%rax,1)              #  246   0x24bec  5      OPC=movb_m8_imm8       
  movl %edi, %edi                       #  247   0x24bf1  2      OPC=movl_r32_r32       
  movl 0x8(%r15,%rdi,1), %eax           #  248   0x24bf3  5      OPC=movl_r32_m32       
  cvtsi2ssq %rax, %xmm0                 #  249   0x24bf8  5      OPC=cvtsi2ssq_xmm_r64  
  mulss 0xfffb9f1(%rip), %xmm0          #  250   0x24bfd  8      OPC=mulss_xmm_m32      
  nop                                   #  251   0x24c05  1      OPC=nop                
  divss -0x44(%rbp), %xmm0              #  252   0x24c06  5      OPC=divss_xmm_m32      
  cvtss2sd %xmm0, %xmm0                 #  253   0x24c0b  4      OPC=cvtss2sd_xmm_xmm   
  movl -0x34(%rbp), %esi                #  254   0x24c0f  3      OPC=movl_r32_m32       
  subl %r12d, %esi                      #  255   0x24c12  3      OPC=subl_r32_r32       
  movl -0x40(%rbp), %edx                #  256   0x24c15  3      OPC=movl_r32_m32       
  leal (%r12,%rdx,1), %edi              #  257   0x24c18  4      OPC=leal_r32_m16       
  movl %ebx, %ecx                       #  258   0x24c1c  2      OPC=movl_r32_r32       
  leal 0xfffb8de(%rip), %edx            #  259   0x24c1e  6      OPC=leal_r32_m16       
  xchgw %ax, %ax                        #  260   0x24c24  2      OPC=xchgw_ax_r16       
  movl $0x1, %eax                       #  261   0x24c26  5      OPC=movl_r32_imm32     
  nop                                   #  262   0x24c2b  1      OPC=nop                
  nop                                   #  263   0x24c2c  1      OPC=nop                
  nop                                   #  264   0x24c2d  1      OPC=nop                
  nop                                   #  265   0x24c2e  1      OPC=nop                
  nop                                   #  266   0x24c2f  1      OPC=nop                
  nop                                   #  267   0x24c30  1      OPC=nop                
  nop                                   #  268   0x24c31  1      OPC=nop                
  nop                                   #  269   0x24c32  1      OPC=nop                
  nop                                   #  270   0x24c33  1      OPC=nop                
  nop                                   #  271   0x24c34  1      OPC=nop                
  nop                                   #  272   0x24c35  1      OPC=nop                
  nop                                   #  273   0x24c36  1      OPC=nop                
  nop                                   #  274   0x24c37  1      OPC=nop                
  nop                                   #  275   0x24c38  1      OPC=nop                
  nop                                   #  276   0x24c39  1      OPC=nop                
  nop                                   #  277   0x24c3a  1      OPC=nop                
  nop                                   #  278   0x24c3b  1      OPC=nop                
  nop                                   #  279   0x24c3c  1      OPC=nop                
  nop                                   #  280   0x24c3d  1      OPC=nop                
  nop                                   #  281   0x24c3e  1      OPC=nop                
  nop                                   #  282   0x24c3f  1      OPC=nop                
  nop                                   #  283   0x24c40  1      OPC=nop                
  callq .snprintf                       #  284   0x24c41  5      OPC=callq_label        
  addl $0x1, %r14d                      #  285   0x24c46  4      OPC=addl_r32_imm8      
  addl $0x10, %r13d                     #  286   0x24c4a  4      OPC=addl_r32_imm8      
  cmpl %r14d, -0x24(%rbp)               #  287   0x24c4e  4      OPC=cmpl_m32_r32       
  jle .L_24c60                          #  288   0x24c52  2      OPC=jle_label          
  addl %eax, %r12d                      #  289   0x24c54  3      OPC=addl_r32_r32       
  jmpq .L_24b80                         #  290   0x24c57  5      OPC=jmpq_label_1       
  nop                                   #  291   0x24c5c  1      OPC=nop                
  nop                                   #  292   0x24c5d  1      OPC=nop                
  nop                                   #  293   0x24c5e  1      OPC=nop                
  nop                                   #  294   0x24c5f  1      OPC=nop                
  nop                                   #  295   0x24c60  1      OPC=nop                
  nop                                   #  296   0x24c61  1      OPC=nop                
  nop                                   #  297   0x24c62  1      OPC=nop                
  nop                                   #  298   0x24c63  1      OPC=nop                
  nop                                   #  299   0x24c64  1      OPC=nop                
  nop                                   #  300   0x24c65  1      OPC=nop                
.L_24c60:                               #        0x24c66  0      OPC=<label>            
  movl -0x50(%rbp), %edi                #  301   0x24c66  3      OPC=movl_r32_m32       
  nop                                   #  302   0x24c69  1      OPC=nop                
  nop                                   #  303   0x24c6a  1      OPC=nop                
  nop                                   #  304   0x24c6b  1      OPC=nop                
  nop                                   #  305   0x24c6c  1      OPC=nop                
  nop                                   #  306   0x24c6d  1      OPC=nop                
  nop                                   #  307   0x24c6e  1      OPC=nop                
  nop                                   #  308   0x24c6f  1      OPC=nop                
  nop                                   #  309   0x24c70  1      OPC=nop                
  nop                                   #  310   0x24c71  1      OPC=nop                
  nop                                   #  311   0x24c72  1      OPC=nop                
  nop                                   #  312   0x24c73  1      OPC=nop                
  nop                                   #  313   0x24c74  1      OPC=nop                
  nop                                   #  314   0x24c75  1      OPC=nop                
  nop                                   #  315   0x24c76  1      OPC=nop                
  nop                                   #  316   0x24c77  1      OPC=nop                
  nop                                   #  317   0x24c78  1      OPC=nop                
  nop                                   #  318   0x24c79  1      OPC=nop                
  nop                                   #  319   0x24c7a  1      OPC=nop                
  nop                                   #  320   0x24c7b  1      OPC=nop                
  nop                                   #  321   0x24c7c  1      OPC=nop                
  nop                                   #  322   0x24c7d  1      OPC=nop                
  nop                                   #  323   0x24c7e  1      OPC=nop                
  nop                                   #  324   0x24c7f  1      OPC=nop                
  nop                                   #  325   0x24c80  1      OPC=nop                
  callq .free                           #  326   0x24c81  5      OPC=callq_label        
  movl -0x58(%rbp), %edi                #  327   0x24c86  3      OPC=movl_r32_m32       
  nop                                   #  328   0x24c89  1      OPC=nop                
  nop                                   #  329   0x24c8a  1      OPC=nop                
  nop                                   #  330   0x24c8b  1      OPC=nop                
  nop                                   #  331   0x24c8c  1      OPC=nop                
  nop                                   #  332   0x24c8d  1      OPC=nop                
  nop                                   #  333   0x24c8e  1      OPC=nop                
  nop                                   #  334   0x24c8f  1      OPC=nop                
  nop                                   #  335   0x24c90  1      OPC=nop                
  nop                                   #  336   0x24c91  1      OPC=nop                
  nop                                   #  337   0x24c92  1      OPC=nop                
  nop                                   #  338   0x24c93  1      OPC=nop                
  nop                                   #  339   0x24c94  1      OPC=nop                
  nop                                   #  340   0x24c95  1      OPC=nop                
  nop                                   #  341   0x24c96  1      OPC=nop                
  nop                                   #  342   0x24c97  1      OPC=nop                
  nop                                   #  343   0x24c98  1      OPC=nop                
  nop                                   #  344   0x24c99  1      OPC=nop                
  nop                                   #  345   0x24c9a  1      OPC=nop                
  nop                                   #  346   0x24c9b  1      OPC=nop                
  nop                                   #  347   0x24c9c  1      OPC=nop                
  nop                                   #  348   0x24c9d  1      OPC=nop                
  nop                                   #  349   0x24c9e  1      OPC=nop                
  nop                                   #  350   0x24c9f  1      OPC=nop                
  nop                                   #  351   0x24ca0  1      OPC=nop                
  callq .ht_destroy                     #  352   0x24ca1  5      OPC=callq_label        
  leal -0x20(%rbp), %esp                #  353   0x24ca6  3      OPC=leal_r32_m16       
  addq %r15, %rsp                       #  354   0x24ca9  3      OPC=addq_r64_r64       
  popq %rbx                             #  355   0x24cac  1      OPC=popq_r64_1         
  popq %r12                             #  356   0x24cad  2      OPC=popq_r64_1         
  popq %r13                             #  357   0x24caf  2      OPC=popq_r64_1         
  popq %r14                             #  358   0x24cb1  2      OPC=popq_r64_1         
  popq %r11                             #  359   0x24cb3  2      OPC=popq_r64_1         
  movl %r11d, %ebp                      #  360   0x24cb5  3      OPC=movl_r32_r32       
  addq %r15, %rbp                       #  361   0x24cb8  3      OPC=addq_r64_r64       
  popq %r11                             #  362   0x24cbb  2      OPC=popq_r64_1         
  nop                                   #  363   0x24cbd  1      OPC=nop                
  nop                                   #  364   0x24cbe  1      OPC=nop                
  nop                                   #  365   0x24cbf  1      OPC=nop                
  nop                                   #  366   0x24cc0  1      OPC=nop                
  nop                                   #  367   0x24cc1  1      OPC=nop                
  nop                                   #  368   0x24cc2  1      OPC=nop                
  nop                                   #  369   0x24cc3  1      OPC=nop                
  nop                                   #  370   0x24cc4  1      OPC=nop                
  nop                                   #  371   0x24cc5  1      OPC=nop                
  andl $0xffffffe0, %r11d               #  372   0x24cc6  7      OPC=andl_r32_imm32     
  nop                                   #  373   0x24ccd  1      OPC=nop                
  nop                                   #  374   0x24cce  1      OPC=nop                
  nop                                   #  375   0x24ccf  1      OPC=nop                
  nop                                   #  376   0x24cd0  1      OPC=nop                
  addq %r15, %r11                       #  377   0x24cd1  3      OPC=addq_r64_r64       
  jmpq %r11                             #  378   0x24cd4  3      OPC=jmpq_r64           
  nop                                   #  379   0x24cd7  1      OPC=nop                
  nop                                   #  380   0x24cd8  1      OPC=nop                
  nop                                   #  381   0x24cd9  1      OPC=nop                
  nop                                   #  382   0x24cda  1      OPC=nop                
  nop                                   #  383   0x24cdb  1      OPC=nop                
  nop                                   #  384   0x24cdc  1      OPC=nop                
  nop                                   #  385   0x24cdd  1      OPC=nop                
  nop                                   #  386   0x24cde  1      OPC=nop                
  nop                                   #  387   0x24cdf  1      OPC=nop                
  nop                                   #  388   0x24ce0  1      OPC=nop                
  nop                                   #  389   0x24ce1  1      OPC=nop                
  nop                                   #  390   0x24ce2  1      OPC=nop                
  nop                                   #  391   0x24ce3  1      OPC=nop                
  nop                                   #  392   0x24ce4  1      OPC=nop                
  nop                                   #  393   0x24ce5  1      OPC=nop                
  nop                                   #  394   0x24ce6  1      OPC=nop                
  nop                                   #  395   0x24ce7  1      OPC=nop                
  nop                                   #  396   0x24ce8  1      OPC=nop                
  nop                                   #  397   0x24ce9  1      OPC=nop                
  nop                                   #  398   0x24cea  1      OPC=nop                
  nop                                   #  399   0x24ceb  1      OPC=nop                
  nop                                   #  400   0x24cec  1      OPC=nop                
                                                                                        
.size print_freqs, .-print_freqs

