  .text
  .globl bulk_free
  .type bulk_free, @function

#! file-offset 0x152cc0
#! rip-offset  0x112cc0
#! capacity    640 bytes

# Text                         #  Line  RIP       Bytes  Opcode              
.bulk_free:                    #        0x112cc0  0      OPC=<label>         
  pushq %r13                   #  1     0x112cc0  2      OPC=pushq_r64_1     
  pushq %r12                   #  2     0x112cc2  2      OPC=pushq_r64_1     
  pushq %rbx                   #  3     0x112cc4  1      OPC=pushq_r64_1     
  movl %edi, %ebx              #  4     0x112cc5  2      OPC=movl_r32_r32    
  subl $0x10, %esp             #  5     0x112cc7  3      OPC=subl_r32_imm8   
  addq %r15, %rsp              #  6     0x112cca  3      OPC=addq_r64_r64    
  testb $0x2, 0xff66068(%rip)  #  7     0x112ccd  7      OPC=testb_m8_imm8   
  jne .L_112e80                #  8     0x112cd4  6      OPC=jne_label_1     
  nop                          #  9     0x112cda  1      OPC=nop             
  nop                          #  10    0x112cdb  1      OPC=nop             
  nop                          #  11    0x112cdc  1      OPC=nop             
  nop                          #  12    0x112cdd  1      OPC=nop             
  nop                          #  13    0x112cde  1      OPC=nop             
  nop                          #  14    0x112cdf  1      OPC=nop             
.L_112ce0:                     #        0x112ce0  0      OPC=<label>         
  leal (%rbx,%rsi,4), %r13d    #  15    0x112ce0  4      OPC=leal_r32_m16    
  cmpl %r13d, %ebx             #  16    0x112ce4  3      OPC=cmpl_r32_r32    
  jne .L_112da0                #  17    0x112ce7  6      OPC=jne_label_1     
  jmpq .L_112de0               #  18    0x112ced  5      OPC=jmpq_label_1    
  nop                          #  19    0x112cf2  1      OPC=nop             
  nop                          #  20    0x112cf3  1      OPC=nop             
  nop                          #  21    0x112cf4  1      OPC=nop             
  nop                          #  22    0x112cf5  1      OPC=nop             
  nop                          #  23    0x112cf6  1      OPC=nop             
  nop                          #  24    0x112cf7  1      OPC=nop             
  nop                          #  25    0x112cf8  1      OPC=nop             
  nop                          #  26    0x112cf9  1      OPC=nop             
  nop                          #  27    0x112cfa  1      OPC=nop             
  nop                          #  28    0x112cfb  1      OPC=nop             
  nop                          #  29    0x112cfc  1      OPC=nop             
  nop                          #  30    0x112cfd  1      OPC=nop             
  nop                          #  31    0x112cfe  1      OPC=nop             
  nop                          #  32    0x112cff  1      OPC=nop             
.L_112d00:                     #        0x112d00  0      OPC=<label>         
  leal -0x8(%rax), %edi        #  33    0x112d00  3      OPC=leal_r32_m16    
  cmpl %edi, 0xff65e87(%rip)   #  34    0x112d03  6      OPC=cmpl_m32_r32    
  movl %edx, %edx              #  35    0x112d09  2      OPC=movl_r32_r32    
  movl $0x0, (%r15,%rdx,1)     #  36    0x112d0b  8      OPC=movl_m32_imm32  
  movl %edi, %edi              #  37    0x112d13  2      OPC=movl_r32_r32    
  movl 0x4(%r15,%rdi,1), %ecx  #  38    0x112d15  5      OPC=movl_r32_m32    
  ja .L_112f20                 #  39    0x112d1a  6      OPC=ja_label_1      
  movl %ecx, %edx              #  40    0x112d20  2      OPC=movl_r32_r32    
  andl $0x3, %edx              #  41    0x112d22  3      OPC=andl_r32_imm8   
  cmpl $0x1, %edx              #  42    0x112d25  3      OPC=cmpl_r32_imm8   
  je .L_112f20                 #  43    0x112d28  6      OPC=je_label_1      
  movl %ecx, %esi              #  44    0x112d2e  2      OPC=movl_r32_r32    
  movl %ebx, %r12d             #  45    0x112d30  3      OPC=movl_r32_r32    
  andl $0xfffffff8, %esi       #  46    0x112d33  6      OPC=andl_r32_imm32  
  nop                          #  47    0x112d39  1      OPC=nop             
  nop                          #  48    0x112d3a  1      OPC=nop             
  nop                          #  49    0x112d3b  1      OPC=nop             
  cmpl %r12d, %r13d            #  50    0x112d3c  3      OPC=cmpl_r32_r32    
  je .L_112d60                 #  51    0x112d3f  2      OPC=je_label        
  leal (%rsi,%rdi,1), %edx     #  52    0x112d41  3      OPC=leal_r32_m16    
  xchgw %ax, %ax               #  53    0x112d44  2      OPC=xchgw_ax_r16    
  leal 0x8(%rdx), %r8d         #  54    0x112d46  4      OPC=leal_r32_m16    
  movl %r12d, %r12d            #  55    0x112d4a  3      OPC=movl_r32_r32    
  cmpl %r8d, (%r15,%r12,1)     #  56    0x112d4d  4      OPC=cmpl_m32_r32    
  je .L_112e40                 #  57    0x112d51  6      OPC=je_label_1      
  nop                          #  58    0x112d57  1      OPC=nop             
  nop                          #  59    0x112d58  1      OPC=nop             
  nop                          #  60    0x112d59  1      OPC=nop             
  nop                          #  61    0x112d5a  1      OPC=nop             
  nop                          #  62    0x112d5b  1      OPC=nop             
  nop                          #  63    0x112d5c  1      OPC=nop             
  nop                          #  64    0x112d5d  1      OPC=nop             
  nop                          #  65    0x112d5e  1      OPC=nop             
  nop                          #  66    0x112d5f  1      OPC=nop             
  nop                          #  67    0x112d60  1      OPC=nop             
  nop                          #  68    0x112d61  1      OPC=nop             
  nop                          #  69    0x112d62  1      OPC=nop             
  nop                          #  70    0x112d63  1      OPC=nop             
  nop                          #  71    0x112d64  1      OPC=nop             
  nop                          #  72    0x112d65  1      OPC=nop             
.L_112d60:                     #        0x112d66  0      OPC=<label>         
  nop                          #  73    0x112d66  1      OPC=nop             
  nop                          #  74    0x112d67  1      OPC=nop             
  nop                          #  75    0x112d68  1      OPC=nop             
  nop                          #  76    0x112d69  1      OPC=nop             
  nop                          #  77    0x112d6a  1      OPC=nop             
  nop                          #  78    0x112d6b  1      OPC=nop             
  nop                          #  79    0x112d6c  1      OPC=nop             
  nop                          #  80    0x112d6d  1      OPC=nop             
  nop                          #  81    0x112d6e  1      OPC=nop             
  nop                          #  82    0x112d6f  1      OPC=nop             
  nop                          #  83    0x112d70  1      OPC=nop             
  nop                          #  84    0x112d71  1      OPC=nop             
  nop                          #  85    0x112d72  1      OPC=nop             
  nop                          #  86    0x112d73  1      OPC=nop             
  nop                          #  87    0x112d74  1      OPC=nop             
  nop                          #  88    0x112d75  1      OPC=nop             
  nop                          #  89    0x112d76  1      OPC=nop             
  nop                          #  90    0x112d77  1      OPC=nop             
  nop                          #  91    0x112d78  1      OPC=nop             
  nop                          #  92    0x112d79  1      OPC=nop             
  nop                          #  93    0x112d7a  1      OPC=nop             
  nop                          #  94    0x112d7b  1      OPC=nop             
  nop                          #  95    0x112d7c  1      OPC=nop             
  nop                          #  96    0x112d7d  1      OPC=nop             
  nop                          #  97    0x112d7e  1      OPC=nop             
  nop                          #  98    0x112d7f  1      OPC=nop             
  nop                          #  99    0x112d80  1      OPC=nop             
  callq .T_267                 #  100   0x112d81  5      OPC=callq_label     
  cmpl %r12d, %r13d            #  101   0x112d86  3      OPC=cmpl_r32_r32    
  je .L_112de0                 #  102   0x112d89  2      OPC=je_label        
  nop                          #  103   0x112d8b  1      OPC=nop             
  nop                          #  104   0x112d8c  1      OPC=nop             
  nop                          #  105   0x112d8d  1      OPC=nop             
  nop                          #  106   0x112d8e  1      OPC=nop             
  nop                          #  107   0x112d8f  1      OPC=nop             
  nop                          #  108   0x112d90  1      OPC=nop             
  nop                          #  109   0x112d91  1      OPC=nop             
  nop                          #  110   0x112d92  1      OPC=nop             
  nop                          #  111   0x112d93  1      OPC=nop             
  nop                          #  112   0x112d94  1      OPC=nop             
  nop                          #  113   0x112d95  1      OPC=nop             
  nop                          #  114   0x112d96  1      OPC=nop             
  nop                          #  115   0x112d97  1      OPC=nop             
  nop                          #  116   0x112d98  1      OPC=nop             
  nop                          #  117   0x112d99  1      OPC=nop             
  nop                          #  118   0x112d9a  1      OPC=nop             
  nop                          #  119   0x112d9b  1      OPC=nop             
  nop                          #  120   0x112d9c  1      OPC=nop             
  nop                          #  121   0x112d9d  1      OPC=nop             
  nop                          #  122   0x112d9e  1      OPC=nop             
  nop                          #  123   0x112d9f  1      OPC=nop             
  nop                          #  124   0x112da0  1      OPC=nop             
  nop                          #  125   0x112da1  1      OPC=nop             
  nop                          #  126   0x112da2  1      OPC=nop             
  nop                          #  127   0x112da3  1      OPC=nop             
  nop                          #  128   0x112da4  1      OPC=nop             
  nop                          #  129   0x112da5  1      OPC=nop             
.L_112da0:                     #        0x112da6  0      OPC=<label>         
  addl $0x4, %ebx              #  130   0x112da6  3      OPC=addl_r32_imm8   
  leal -0x4(%rbx), %edx        #  131   0x112da9  3      OPC=leal_r32_m16    
  movl %edx, %edx              #  132   0x112dac  2      OPC=movl_r32_r32    
  movl (%r15,%rdx,1), %eax     #  133   0x112dae  4      OPC=movl_r32_m32    
  testq %rax, %rax             #  134   0x112db2  3      OPC=testq_r64_r64   
  jne .L_112d00                #  135   0x112db5  6      OPC=jne_label_1     
  movl %ebx, %r12d             #  136   0x112dbb  3      OPC=movl_r32_r32    
  cmpl %r12d, %r13d            #  137   0x112dbe  3      OPC=cmpl_r32_r32    
  nop                          #  138   0x112dc1  1      OPC=nop             
  nop                          #  139   0x112dc2  1      OPC=nop             
  nop                          #  140   0x112dc3  1      OPC=nop             
  nop                          #  141   0x112dc4  1      OPC=nop             
  nop                          #  142   0x112dc5  1      OPC=nop             
  jne .L_112da0                #  143   0x112dc6  2      OPC=jne_label       
  nop                          #  144   0x112dc8  1      OPC=nop             
  nop                          #  145   0x112dc9  1      OPC=nop             
  nop                          #  146   0x112dca  1      OPC=nop             
  nop                          #  147   0x112dcb  1      OPC=nop             
  nop                          #  148   0x112dcc  1      OPC=nop             
  nop                          #  149   0x112dcd  1      OPC=nop             
  nop                          #  150   0x112dce  1      OPC=nop             
  nop                          #  151   0x112dcf  1      OPC=nop             
  nop                          #  152   0x112dd0  1      OPC=nop             
  nop                          #  153   0x112dd1  1      OPC=nop             
  nop                          #  154   0x112dd2  1      OPC=nop             
  nop                          #  155   0x112dd3  1      OPC=nop             
  nop                          #  156   0x112dd4  1      OPC=nop             
  nop                          #  157   0x112dd5  1      OPC=nop             
  nop                          #  158   0x112dd6  1      OPC=nop             
  nop                          #  159   0x112dd7  1      OPC=nop             
  nop                          #  160   0x112dd8  1      OPC=nop             
  nop                          #  161   0x112dd9  1      OPC=nop             
  nop                          #  162   0x112dda  1      OPC=nop             
  nop                          #  163   0x112ddb  1      OPC=nop             
  nop                          #  164   0x112ddc  1      OPC=nop             
  nop                          #  165   0x112ddd  1      OPC=nop             
  nop                          #  166   0x112dde  1      OPC=nop             
  nop                          #  167   0x112ddf  1      OPC=nop             
  nop                          #  168   0x112de0  1      OPC=nop             
  nop                          #  169   0x112de1  1      OPC=nop             
  nop                          #  170   0x112de2  1      OPC=nop             
  nop                          #  171   0x112de3  1      OPC=nop             
  nop                          #  172   0x112de4  1      OPC=nop             
  nop                          #  173   0x112de5  1      OPC=nop             
.L_112de0:                     #        0x112de6  0      OPC=<label>         
  movl 0xff65da6(%rip), %eax   #  174   0x112de6  6      OPC=movl_r32_m32    
  cmpl 0xff65db0(%rip), %eax   #  175   0x112dec  6      OPC=cmpl_r32_m32    
  ja .L_112ee0                 #  176   0x112df2  6      OPC=ja_label_1      
  nop                          #  177   0x112df8  1      OPC=nop             
  nop                          #  178   0x112df9  1      OPC=nop             
  nop                          #  179   0x112dfa  1      OPC=nop             
  nop                          #  180   0x112dfb  1      OPC=nop             
  nop                          #  181   0x112dfc  1      OPC=nop             
  nop                          #  182   0x112dfd  1      OPC=nop             
  nop                          #  183   0x112dfe  1      OPC=nop             
  nop                          #  184   0x112dff  1      OPC=nop             
  nop                          #  185   0x112e00  1      OPC=nop             
  nop                          #  186   0x112e01  1      OPC=nop             
  nop                          #  187   0x112e02  1      OPC=nop             
  nop                          #  188   0x112e03  1      OPC=nop             
  nop                          #  189   0x112e04  1      OPC=nop             
  nop                          #  190   0x112e05  1      OPC=nop             
.L_112e00:                     #        0x112e06  0      OPC=<label>         
  testb $0x2, 0xff65f35(%rip)  #  191   0x112e06  7      OPC=testb_m8_imm8   
  je .L_112e20                 #  192   0x112e0d  2      OPC=je_label        
  mfence                       #  193   0x112e0f  3      OPC=mfence          
  movl $0x0, 0xff65f2a(%rip)   #  194   0x112e12  10     OPC=movl_m32_imm32  
  nop                          #  195   0x112e1c  1      OPC=nop             
  nop                          #  196   0x112e1d  1      OPC=nop             
  nop                          #  197   0x112e1e  1      OPC=nop             
  nop                          #  198   0x112e1f  1      OPC=nop             
  nop                          #  199   0x112e20  1      OPC=nop             
  nop                          #  200   0x112e21  1      OPC=nop             
  nop                          #  201   0x112e22  1      OPC=nop             
  nop                          #  202   0x112e23  1      OPC=nop             
  nop                          #  203   0x112e24  1      OPC=nop             
  nop                          #  204   0x112e25  1      OPC=nop             
.L_112e20:                     #        0x112e26  0      OPC=<label>         
  addl $0x10, %esp             #  205   0x112e26  3      OPC=addl_r32_imm8   
  addq %r15, %rsp              #  206   0x112e29  3      OPC=addq_r64_r64    
  xorl %eax, %eax              #  207   0x112e2c  2      OPC=xorl_r32_r32    
  popq %rbx                    #  208   0x112e2e  1      OPC=popq_r64_1      
  popq %r12                    #  209   0x112e2f  2      OPC=popq_r64_1      
  popq %r13                    #  210   0x112e31  2      OPC=popq_r64_1      
  popq %r11                    #  211   0x112e33  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d      #  212   0x112e35  7      OPC=andl_r32_imm32  
  nop                          #  213   0x112e3c  1      OPC=nop             
  nop                          #  214   0x112e3d  1      OPC=nop             
  nop                          #  215   0x112e3e  1      OPC=nop             
  nop                          #  216   0x112e3f  1      OPC=nop             
  addq %r15, %r11              #  217   0x112e40  3      OPC=addq_r64_r64    
  jmpq %r11                    #  218   0x112e43  3      OPC=jmpq_r64        
  nop                          #  219   0x112e46  1      OPC=nop             
  nop                          #  220   0x112e47  1      OPC=nop             
  nop                          #  221   0x112e48  1      OPC=nop             
  nop                          #  222   0x112e49  1      OPC=nop             
  nop                          #  223   0x112e4a  1      OPC=nop             
  nop                          #  224   0x112e4b  1      OPC=nop             
  nop                          #  225   0x112e4c  1      OPC=nop             
.L_112e40:                     #        0x112e4d  0      OPC=<label>         
  movl %edx, %edx              #  226   0x112e4d  2      OPC=movl_r32_r32    
  movl 0x4(%r15,%rdx,1), %edx  #  227   0x112e4f  5      OPC=movl_r32_m32    
  andl $0x1, %ecx              #  228   0x112e54  3      OPC=andl_r32_imm8   
  movl %r12d, %r12d            #  229   0x112e57  3      OPC=movl_r32_r32    
  movl %eax, (%r15,%r12,1)     #  230   0x112e5a  4      OPC=movl_m32_r32    
  orl $0x2, %ecx               #  231   0x112e5e  3      OPC=orl_r32_imm8    
  andl $0xfffffff8, %edx       #  232   0x112e61  6      OPC=andl_r32_imm32  
  nop                          #  233   0x112e67  1      OPC=nop             
  nop                          #  234   0x112e68  1      OPC=nop             
  nop                          #  235   0x112e69  1      OPC=nop             
  leal (%rdx,%rsi,1), %esi     #  236   0x112e6a  3      OPC=leal_r32_m16    
  orl %esi, %ecx               #  237   0x112e6d  2      OPC=orl_r32_r32     
  nop                          #  238   0x112e6f  1      OPC=nop             
  nop                          #  239   0x112e70  1      OPC=nop             
  nop                          #  240   0x112e71  1      OPC=nop             
  nop                          #  241   0x112e72  1      OPC=nop             
  movl %edi, %edi              #  242   0x112e73  2      OPC=movl_r32_r32    
  movl %ecx, 0x4(%r15,%rdi,1)  #  243   0x112e75  5      OPC=movl_m32_r32    
  leal (%rsi,%rdi,1), %edi     #  244   0x112e7a  3      OPC=leal_r32_m16    
  movl %edi, %edi              #  245   0x112e7d  2      OPC=movl_r32_r32    
  orl $0x1, 0x4(%r15,%rdi,1)   #  246   0x112e7f  6      OPC=orl_m32_imm8    
  jmpq .L_112da0               #  247   0x112e85  5      OPC=jmpq_label_1    
  nop                          #  248   0x112e8a  1      OPC=nop             
  nop                          #  249   0x112e8b  1      OPC=nop             
  nop                          #  250   0x112e8c  1      OPC=nop             
  nop                          #  251   0x112e8d  1      OPC=nop             
  nop                          #  252   0x112e8e  1      OPC=nop             
  nop                          #  253   0x112e8f  1      OPC=nop             
  nop                          #  254   0x112e90  1      OPC=nop             
  nop                          #  255   0x112e91  1      OPC=nop             
  nop                          #  256   0x112e92  1      OPC=nop             
.L_112e80:                     #        0x112e93  0      OPC=<label>         
  movl $0x1, %eax              #  257   0x112e93  5      OPC=movl_r32_imm32  
  xchgl %eax, 0xff65eb5(%rip)  #  258   0x112e98  6      OPC=xchgl_m32_r32   
  testl %eax, %eax             #  259   0x112e9e  2      OPC=testl_r32_r32   
  je .L_112ce0                 #  260   0x112ea0  6      OPC=je_label_1      
  movl $0x10078d40, %edi       #  261   0x112ea6  5      OPC=movl_r32_imm32  
  movl %esi, 0x8(%rsp)         #  262   0x112eab  4      OPC=movl_m32_r32    
  nop                          #  263   0x112eaf  1      OPC=nop             
  nop                          #  264   0x112eb0  1      OPC=nop             
  nop                          #  265   0x112eb1  1      OPC=nop             
  nop                          #  266   0x112eb2  1      OPC=nop             
  nop                          #  267   0x112eb3  1      OPC=nop             
  nop                          #  268   0x112eb4  1      OPC=nop             
  nop                          #  269   0x112eb5  1      OPC=nop             
  nop                          #  270   0x112eb6  1      OPC=nop             
  nop                          #  271   0x112eb7  1      OPC=nop             
  nop                          #  272   0x112eb8  1      OPC=nop             
  nop                          #  273   0x112eb9  1      OPC=nop             
  nop                          #  274   0x112eba  1      OPC=nop             
  nop                          #  275   0x112ebb  1      OPC=nop             
  nop                          #  276   0x112ebc  1      OPC=nop             
  nop                          #  277   0x112ebd  1      OPC=nop             
  nop                          #  278   0x112ebe  1      OPC=nop             
  nop                          #  279   0x112ebf  1      OPC=nop             
  nop                          #  280   0x112ec0  1      OPC=nop             
  nop                          #  281   0x112ec1  1      OPC=nop             
  nop                          #  282   0x112ec2  1      OPC=nop             
  nop                          #  283   0x112ec3  1      OPC=nop             
  nop                          #  284   0x112ec4  1      OPC=nop             
  nop                          #  285   0x112ec5  1      OPC=nop             
  nop                          #  286   0x112ec6  1      OPC=nop             
  nop                          #  287   0x112ec7  1      OPC=nop             
  nop                          #  288   0x112ec8  1      OPC=nop             
  nop                          #  289   0x112ec9  1      OPC=nop             
  nop                          #  290   0x112eca  1      OPC=nop             
  nop                          #  291   0x112ecb  1      OPC=nop             
  nop                          #  292   0x112ecc  1      OPC=nop             
  nop                          #  293   0x112ecd  1      OPC=nop             
  callq .spin_acquire_lock     #  294   0x112ece  5      OPC=callq_label     
  testl %eax, %eax             #  295   0x112ed3  2      OPC=testl_r32_r32   
  movl 0x8(%rsp), %esi         #  296   0x112ed5  4      OPC=movl_r32_m32    
  je .L_112ce0                 #  297   0x112ed9  6      OPC=je_label_1      
  jmpq .L_112e20               #  298   0x112edf  5      OPC=jmpq_label_1    
  nop                          #  299   0x112ee4  1      OPC=nop             
  nop                          #  300   0x112ee5  1      OPC=nop             
  nop                          #  301   0x112ee6  1      OPC=nop             
  nop                          #  302   0x112ee7  1      OPC=nop             
  nop                          #  303   0x112ee8  1      OPC=nop             
  nop                          #  304   0x112ee9  1      OPC=nop             
  nop                          #  305   0x112eea  1      OPC=nop             
  nop                          #  306   0x112eeb  1      OPC=nop             
  nop                          #  307   0x112eec  1      OPC=nop             
  nop                          #  308   0x112eed  1      OPC=nop             
  nop                          #  309   0x112eee  1      OPC=nop             
  nop                          #  310   0x112eef  1      OPC=nop             
  nop                          #  311   0x112ef0  1      OPC=nop             
  nop                          #  312   0x112ef1  1      OPC=nop             
  nop                          #  313   0x112ef2  1      OPC=nop             
.L_112ee0:                     #        0x112ef3  0      OPC=<label>         
  xorl %edi, %edi              #  314   0x112ef3  2      OPC=xorl_r32_r32    
  nop                          #  315   0x112ef5  1      OPC=nop             
  nop                          #  316   0x112ef6  1      OPC=nop             
  nop                          #  317   0x112ef7  1      OPC=nop             
  nop                          #  318   0x112ef8  1      OPC=nop             
  nop                          #  319   0x112ef9  1      OPC=nop             
  nop                          #  320   0x112efa  1      OPC=nop             
  nop                          #  321   0x112efb  1      OPC=nop             
  nop                          #  322   0x112efc  1      OPC=nop             
  nop                          #  323   0x112efd  1      OPC=nop             
  nop                          #  324   0x112efe  1      OPC=nop             
  nop                          #  325   0x112eff  1      OPC=nop             
  nop                          #  326   0x112f00  1      OPC=nop             
  nop                          #  327   0x112f01  1      OPC=nop             
  nop                          #  328   0x112f02  1      OPC=nop             
  nop                          #  329   0x112f03  1      OPC=nop             
  nop                          #  330   0x112f04  1      OPC=nop             
  nop                          #  331   0x112f05  1      OPC=nop             
  nop                          #  332   0x112f06  1      OPC=nop             
  nop                          #  333   0x112f07  1      OPC=nop             
  nop                          #  334   0x112f08  1      OPC=nop             
  nop                          #  335   0x112f09  1      OPC=nop             
  nop                          #  336   0x112f0a  1      OPC=nop             
  nop                          #  337   0x112f0b  1      OPC=nop             
  nop                          #  338   0x112f0c  1      OPC=nop             
  nop                          #  339   0x112f0d  1      OPC=nop             
  callq .T_266                 #  340   0x112f0e  5      OPC=callq_label     
  jmpq .L_112e00               #  341   0x112f13  5      OPC=jmpq_label_1    
  nop                          #  342   0x112f18  1      OPC=nop             
  nop                          #  343   0x112f19  1      OPC=nop             
  nop                          #  344   0x112f1a  1      OPC=nop             
  nop                          #  345   0x112f1b  1      OPC=nop             
  nop                          #  346   0x112f1c  1      OPC=nop             
  nop                          #  347   0x112f1d  1      OPC=nop             
  nop                          #  348   0x112f1e  1      OPC=nop             
  nop                          #  349   0x112f1f  1      OPC=nop             
  nop                          #  350   0x112f20  1      OPC=nop             
  nop                          #  351   0x112f21  1      OPC=nop             
  nop                          #  352   0x112f22  1      OPC=nop             
  nop                          #  353   0x112f23  1      OPC=nop             
  nop                          #  354   0x112f24  1      OPC=nop             
  nop                          #  355   0x112f25  1      OPC=nop             
  nop                          #  356   0x112f26  1      OPC=nop             
  nop                          #  357   0x112f27  1      OPC=nop             
  nop                          #  358   0x112f28  1      OPC=nop             
  nop                          #  359   0x112f29  1      OPC=nop             
  nop                          #  360   0x112f2a  1      OPC=nop             
  nop                          #  361   0x112f2b  1      OPC=nop             
  nop                          #  362   0x112f2c  1      OPC=nop             
  nop                          #  363   0x112f2d  1      OPC=nop             
  nop                          #  364   0x112f2e  1      OPC=nop             
  nop                          #  365   0x112f2f  1      OPC=nop             
  nop                          #  366   0x112f30  1      OPC=nop             
  nop                          #  367   0x112f31  1      OPC=nop             
  nop                          #  368   0x112f32  1      OPC=nop             
.L_112f20:                     #        0x112f33  0      OPC=<label>         
  nop                          #  369   0x112f33  1      OPC=nop             
  nop                          #  370   0x112f34  1      OPC=nop             
  nop                          #  371   0x112f35  1      OPC=nop             
  nop                          #  372   0x112f36  1      OPC=nop             
  nop                          #  373   0x112f37  1      OPC=nop             
  nop                          #  374   0x112f38  1      OPC=nop             
  nop                          #  375   0x112f39  1      OPC=nop             
  nop                          #  376   0x112f3a  1      OPC=nop             
  nop                          #  377   0x112f3b  1      OPC=nop             
  nop                          #  378   0x112f3c  1      OPC=nop             
  nop                          #  379   0x112f3d  1      OPC=nop             
  nop                          #  380   0x112f3e  1      OPC=nop             
  nop                          #  381   0x112f3f  1      OPC=nop             
  nop                          #  382   0x112f40  1      OPC=nop             
  nop                          #  383   0x112f41  1      OPC=nop             
  nop                          #  384   0x112f42  1      OPC=nop             
  nop                          #  385   0x112f43  1      OPC=nop             
  nop                          #  386   0x112f44  1      OPC=nop             
  nop                          #  387   0x112f45  1      OPC=nop             
  nop                          #  388   0x112f46  1      OPC=nop             
  nop                          #  389   0x112f47  1      OPC=nop             
  nop                          #  390   0x112f48  1      OPC=nop             
  nop                          #  391   0x112f49  1      OPC=nop             
  nop                          #  392   0x112f4a  1      OPC=nop             
  nop                          #  393   0x112f4b  1      OPC=nop             
  nop                          #  394   0x112f4c  1      OPC=nop             
  nop                          #  395   0x112f4d  1      OPC=nop             
  callq .abort                 #  396   0x112f4e  5      OPC=callq_label     
                                                                             
.size bulk_free, .-bulk_free

