  .text
  .globl rfc3484_sort
  .type rfc3484_sort, @function

#! file-offset 0xcc95b
#! rip-offset  0xcc95b
#! capacity    1437 bytes

# Text                          #  Line  RIP      Bytes  Opcode                
.rfc3484_sort:                  #        0xcc95b  0      OPC=<label>           
  pushq %r15                    #  1     0xcc95b  2      OPC=pushq_r64_1       
  pushq %r14                    #  2     0xcc95d  2      OPC=pushq_r64_1       
  pushq %r13                    #  3     0xcc95f  2      OPC=pushq_r64_1       
  pushq %r12                    #  4     0xcc961  2      OPC=pushq_r64_1       
  pushq %rbp                    #  5     0xcc963  1      OPC=pushq_r64_1       
  pushq %rbx                    #  6     0xcc964  1      OPC=pushq_r64_1       
  subq $0x38, %rsp              #  7     0xcc965  4      OPC=subq_r64_imm8     
  movq %rdx, %r12               #  8     0xcc969  3      OPC=movq_r64_r64      
  movq (%rdi), %r13             #  9     0xcc96c  3      OPC=movq_r64_m64      
  movq (%rsi), %r14             #  10    0xcc96f  3      OPC=movq_r64_m64      
  movq (%rdx), %rax             #  11    0xcc972  3      OPC=movq_r64_m64      
  leaq (%r13,%r13,2), %rbp      #  12    0xcc975  5      OPC=leaq_r64_m16      
  shlq $0x4, %rbp               #  13    0xcc97a  4      OPC=shlq_r64_imm8     
  addq %rax, %rbp               #  14    0xcc97e  3      OPC=addq_r64_r64      
  leaq (%r14,%r14,2), %rbx      #  15    0xcc981  4      OPC=leaq_r64_m16      
  shlq $0x4, %rbx               #  16    0xcc985  4      OPC=shlq_r64_imm8     
  addq %rax, %rbx               #  17    0xcc989  3      OPC=addq_r64_r64      
  cmpb $0x0, 0x25(%rbp)         #  18    0xcc98c  4      OPC=cmpb_m8_imm8      
  je .L_cc9a3                   #  19    0xcc990  2      OPC=je_label          
  movl $0xffffffff, %eax        #  20    0xcc992  6      OPC=movl_r32_imm32_1  
  cmpb $0x0, 0x25(%rbx)         #  21    0xcc998  4      OPC=cmpb_m8_imm8      
  je .L_ccee9                   #  22    0xcc99c  6      OPC=je_label_1        
  jmpq .L_cc9b2                 #  23    0xcc9a2  2      OPC=jmpq_label        
.L_cc9a3:                       #        0xcc9a4  0      OPC=<label>           
  cmpb $0x0, 0x25(%rbx)         #  24    0xcc9a4  4      OPC=cmpb_m8_imm8      
  je .L_cce87                   #  25    0xcc9a8  6      OPC=je_label_1        
  jmpq .L_cce1e                 #  26    0xcc9ae  5      OPC=jmpq_label_1      
.L_cc9b2:                       #        0xcc9b3  0      OPC=<label>           
  movq (%rbp), %rax             #  27    0xcc9b3  4      OPC=movq_r64_m64      
  movq 0x18(%rax), %r15         #  28    0xcc9b7  4      OPC=movq_r64_m64      
  movq %r15, %rdi               #  29    0xcc9bb  3      OPC=movq_r64_r64      
  callq .get_scope              #  30    0xcc9be  5      OPC=callq_label       
  movl %eax, 0x8(%rsp)          #  31    0xcc9c3  4      OPC=movl_m32_r32      
  movq (%rbx), %rax             #  32    0xcc9c7  3      OPC=movq_r64_m64      
  movq 0x18(%rax), %rdi         #  33    0xcc9ca  4      OPC=movq_r64_m64      
  callq .get_scope              #  34    0xcc9ce  5      OPC=callq_label       
  movl %eax, 0x4(%rsp)          #  35    0xcc9d3  4      OPC=movl_m32_r32      
  leaq 0x8(%rbp), %rcx          #  36    0xcc9d7  4      OPC=leaq_r64_m16      
  movq %rcx, 0x10(%rsp)         #  37    0xcc9db  5      OPC=movq_m64_r64      
  movq %rcx, %rdi               #  38    0xcc9e0  3      OPC=movq_r64_r64      
  callq .get_scope              #  39    0xcc9e3  5      OPC=callq_label       
  movl %eax, 0xc(%rsp)          #  40    0xcc9e8  4      OPC=movl_m32_r32      
  leaq 0x8(%rbx), %rcx          #  41    0xcc9ec  4      OPC=leaq_r64_m16      
  movq %rcx, 0x18(%rsp)         #  42    0xcc9f0  5      OPC=movq_m64_r64      
  movq %rcx, %rdi               #  43    0xcc9f5  3      OPC=movq_r64_r64      
  callq .get_scope              #  44    0xcc9f8  5      OPC=callq_label       
  cmpl %eax, 0x4(%rsp)          #  45    0xcc9fd  4      OPC=cmpl_m32_r32      
  je .L_cca10                   #  46    0xcca01  2      OPC=je_label          
  movl 0xc(%rsp), %edx          #  47    0xcca03  4      OPC=movl_r32_m32      
  cmpl %edx, 0x8(%rsp)          #  48    0xcca07  4      OPC=cmpl_m32_r32      
  je .L_cce28                   #  49    0xcca0b  6      OPC=je_label_1        
.L_cca10:                       #        0xcca11  0      OPC=<label>           
  cmpl %eax, 0x4(%rsp)          #  50    0xcca11  4      OPC=cmpl_m32_r32      
  jne .L_cca29                  #  51    0xcca15  2      OPC=jne_label         
  movl $0x1, %eax               #  52    0xcca17  5      OPC=movl_r32_imm32    
  movl 0xc(%rsp), %edx          #  53    0xcca1c  4      OPC=movl_r32_m32      
  cmpl %edx, 0x8(%rsp)          #  54    0xcca20  4      OPC=cmpl_m32_r32      
  jne .L_ccee9                  #  55    0xcca24  6      OPC=jne_label_1       
.L_cca29:                       #        0xcca2a  0      OPC=<label>           
  movzbl 0x26(%rbp), %eax       #  56    0xcca2a  4      OPC=movzbl_r32_m8     
  testb $0x1, %al               #  57    0xcca2e  2      OPC=testb_al_imm8     
  jne .L_cca3c                  #  58    0xcca30  2      OPC=jne_label         
  testb $0x1, 0x26(%rbx)        #  59    0xcca32  4      OPC=testb_m8_imm8     
  je .L_cca46                   #  60    0xcca36  2      OPC=je_label          
  jmpq .L_cce32                 #  61    0xcca38  5      OPC=jmpq_label_1      
.L_cca3c:                       #        0xcca3d  0      OPC=<label>           
  testb $0x1, 0x26(%rbx)        #  62    0xcca3d  4      OPC=testb_m8_imm8     
  je .L_cce3c                   #  63    0xcca41  6      OPC=je_label_1        
.L_cca46:                       #        0xcca47  0      OPC=<label>           
  testb $0x2, %al               #  64    0xcca47  2      OPC=testb_al_imm8     
  jne .L_cca55                  #  65    0xcca49  2      OPC=jne_label         
  testb $0x2, 0x26(%rbx)        #  66    0xcca4b  4      OPC=testb_m8_imm8     
  je .L_cca5f                   #  67    0xcca4f  2      OPC=je_label          
  jmpq .L_cce46                 #  68    0xcca51  5      OPC=jmpq_label_1      
.L_cca55:                       #        0xcca56  0      OPC=<label>           
  testb $0x2, 0x26(%rbx)        #  69    0xcca56  4      OPC=testb_m8_imm8     
  je .L_cce50                   #  70    0xcca5a  6      OPC=je_label_1        
.L_cca5f:                       #        0xcca60  0      OPC=<label>           
  movq %r15, %rdi               #  71    0xcca60  3      OPC=movq_r64_r64      
  callq .get_label              #  72    0xcca63  5      OPC=callq_label       
  movl %eax, 0xc(%rsp)          #  73    0xcca68  4      OPC=movl_m32_r32      
  movq 0x10(%rsp), %rdi         #  74    0xcca6c  5      OPC=movq_r64_m64      
  callq .get_label              #  75    0xcca71  5      OPC=callq_label       
  movl %eax, 0x10(%rsp)         #  76    0xcca76  4      OPC=movl_m32_r32      
  movq (%rbx), %rax             #  77    0xcca7a  3      OPC=movq_r64_m64      
  movq 0x18(%rax), %rdi         #  78    0xcca7d  4      OPC=movq_r64_m64      
  callq .get_label              #  79    0xcca81  5      OPC=callq_label       
  movl %eax, %r15d              #  80    0xcca86  3      OPC=movl_r32_r32      
  movq 0x18(%rsp), %rdi         #  81    0xcca89  5      OPC=movq_r64_m64      
  callq .get_label              #  82    0xcca8e  5      OPC=callq_label       
  cmpl %eax, %r15d              #  83    0xcca93  3      OPC=cmpl_r32_r32      
  je .L_ccaa5                   #  84    0xcca96  2      OPC=je_label          
  movl 0x10(%rsp), %ecx         #  85    0xcca98  4      OPC=movl_r32_m32      
  cmpl %ecx, 0xc(%rsp)          #  86    0xcca9c  4      OPC=cmpl_m32_r32      
  je .L_cce5a                   #  87    0xccaa0  6      OPC=je_label_1        
.L_ccaa5:                       #        0xccaa6  0      OPC=<label>           
  cmpl %eax, %r15d              #  88    0xccaa6  3      OPC=cmpl_r32_r32      
  jne .L_ccabd                  #  89    0xccaa9  2      OPC=jne_label         
  movl $0x1, %eax               #  90    0xccaab  5      OPC=movl_r32_imm32    
  movl 0x10(%rsp), %esi         #  91    0xccab0  4      OPC=movl_r32_m32      
  cmpl %esi, 0xc(%rsp)          #  92    0xccab4  4      OPC=cmpl_m32_r32      
  jne .L_ccee9                  #  93    0xccab8  6      OPC=jne_label_1       
.L_ccabd:                       #        0xccabe  0      OPC=<label>           
  movq 0x2c1764(%rip), %r15     #  94    0xccabe  7      OPC=movq_r64_m64      
  movq (%rbp), %rax             #  95    0xccac5  4      OPC=movq_r64_m64      
  movq 0x18(%rax), %rdi         #  96    0xccac9  4      OPC=movq_r64_m64      
  movl $0x0, %edx               #  97    0xccacd  5      OPC=movl_r32_imm32    
  movq %r15, %rsi               #  98    0xccad2  3      OPC=movq_r64_r64      
  callq .match_prefix           #  99    0xccad5  5      OPC=callq_label       
  movl %eax, 0xc(%rsp)          #  100   0xccada  4      OPC=movl_m32_r32      
  movq (%rbx), %rax             #  101   0xccade  3      OPC=movq_r64_m64      
  movq 0x18(%rax), %rdi         #  102   0xccae1  4      OPC=movq_r64_m64      
  movl $0x0, %edx               #  103   0xccae5  5      OPC=movl_r32_imm32    
  movq %r15, %rsi               #  104   0xccaea  3      OPC=movq_r64_r64      
  callq .match_prefix           #  105   0xccaed  5      OPC=callq_label       
  movl 0xc(%rsp), %esi          #  106   0xccaf2  4      OPC=movl_r32_m32      
  cmpl %eax, %esi               #  107   0xccaf6  2      OPC=cmpl_r32_r32      
  jg .L_cce64                   #  108   0xccaf8  6      OPC=jg_label_1        
  jl .L_cce6b                   #  109   0xccafe  6      OPC=jl_label_1        
  cmpb $0x0, 0x25(%rbp)         #  110   0xccb04  4      OPC=cmpb_m8_imm8      
  je .L_ccc6f                   #  111   0xccb08  6      OPC=je_label_1        
  movl 0x28(%rbp), %r15d        #  112   0xccb0e  4      OPC=movl_r32_m32      
  movl 0x28(%rbx), %eax         #  113   0xccb12  3      OPC=movl_r32_m32      
  movl %eax, 0xc(%rsp)          #  114   0xccb15  4      OPC=movl_m32_r32      
  cmpl %eax, %r15d              #  115   0xccb19  3      OPC=cmpl_r32_r32      
  je .L_ccc6f                   #  116   0xccb1c  6      OPC=je_label_1        
  movl 0x2c(%rbp), %edx         #  117   0xccb22  3      OPC=movl_r32_m32      
  movl %edx, 0x28(%rsp)         #  118   0xccb25  4      OPC=movl_m32_r32      
  movl 0x2c(%rbx), %eax         #  119   0xccb29  3      OPC=movl_r32_m32      
  movl %eax, 0x2c(%rsp)         #  120   0xccb2c  4      OPC=movl_m32_r32      
  cmpl $0xffffffff, %edx        #  121   0xccb30  6      OPC=cmpl_r32_imm32    
  nop                           #  122   0xccb36  1      OPC=nop               
  nop                           #  123   0xccb37  1      OPC=nop               
  nop                           #  124   0xccb38  1      OPC=nop               
  sete %dl                      #  125   0xccb39  3      OPC=sete_r8           
  cmpl $0xffffffff, %eax        #  126   0xccb3c  6      OPC=cmpl_r32_imm32    
  nop                           #  127   0xccb42  1      OPC=nop               
  nop                           #  128   0xccb43  1      OPC=nop               
  nop                           #  129   0xccb44  1      OPC=nop               
  sete %al                      #  130   0xccb45  3      OPC=sete_r8           
  je .L_ccb45                   #  131   0xccb48  2      OPC=je_label          
  testb %dl, %dl                #  132   0xccb4a  2      OPC=testb_r8_r8       
  je .L_ccc42                   #  133   0xccb4c  6      OPC=je_label_1        
.L_ccb45:                       #        0xccb52  0      OPC=<label>           
  testb %dl, %dl                #  134   0xccb52  2      OPC=testb_r8_r8       
  je .L_ccb53                   #  135   0xccb54  2      OPC=je_label          
  movl $0x0, 0x28(%rsp)         #  136   0xccb56  8      OPC=movl_m32_imm32    
  jmpq .L_ccb59                 #  137   0xccb5e  2      OPC=jmpq_label        
.L_ccb53:                       #        0xccb60  0      OPC=<label>           
  movl $0xffffffff, %r15d       #  138   0xccb60  7      OPC=movl_r32_imm32_1  
.L_ccb59:                       #        0xccb67  0      OPC=<label>           
  testb %al, %al                #  139   0xccb67  2      OPC=testb_r8_r8       
  je .L_ccb67                   #  140   0xccb69  2      OPC=je_label          
  movl $0x0, 0x2c(%rsp)         #  141   0xccb6b  8      OPC=movl_m32_imm32    
  jmpq .L_ccb6f                 #  142   0xccb73  2      OPC=jmpq_label        
.L_ccb67:                       #        0xccb75  0      OPC=<label>           
  movl $0xffffffff, 0xc(%rsp)   #  143   0xccb75  8      OPC=movl_m32_imm32    
.L_ccb6f:                       #        0xccb7d  0      OPC=<label>           
  leaq 0x2c(%rsp), %rcx         #  144   0xccb7d  5      OPC=leaq_r64_m16      
  leaq 0x28(%rsp), %rsi         #  145   0xccb82  5      OPC=leaq_r64_m16      
  movl 0xc(%rsp), %edx          #  146   0xccb87  4      OPC=movl_r32_m32      
  movl %r15d, %edi              #  147   0xccb8b  3      OPC=movl_r32_r32      
  callq .__check_native         #  148   0xccb8e  5      OPC=callq_label       
  cmpl $0x0, 0x8(%r12)          #  149   0xccb93  6      OPC=cmpl_m32_imm8     
  jle .L_ccc42                  #  150   0xccb99  6      OPC=jle_label_1       
  movl 0x28(%rsp), %ecx         #  151   0xccb9f  4      OPC=movl_r32_m32      
  movl 0x2c(%rsp), %esi         #  152   0xccba3  4      OPC=movl_r32_m32      
  movl $0x0, %edx               #  153   0xccba7  5      OPC=movl_r32_imm32    
.L_ccb9e:                       #        0xccbac  0      OPC=<label>           
  cmpl $0xffffffff, %r15d       #  154   0xccbac  7      OPC=cmpl_r32_imm32    
  nop                           #  155   0xccbb3  1      OPC=nop               
  nop                           #  156   0xccbb4  1      OPC=nop               
  nop                           #  157   0xccbb5  1      OPC=nop               
  nop                           #  158   0xccbb6  1      OPC=nop               
  je .L_ccbe9                   #  159   0xccbb7  2      OPC=je_label          
  movslq %edx, %rax             #  160   0xccbb9  3      OPC=movslq_r64_r32    
  leaq (%rax,%rax,2), %rax      #  161   0xccbbc  4      OPC=leaq_r64_m16      
  shlq $0x4, %rax               #  162   0xccbc0  4      OPC=shlq_r64_imm8     
  addq (%r12), %rax             #  163   0xccbc4  4      OPC=addq_r64_m64      
  cmpl %r15d, 0x28(%rax)        #  164   0xccbc8  4      OPC=cmpl_m32_r32      
  jne .L_ccbe9                  #  165   0xccbcc  2      OPC=jne_label         
  movl 0x2c(%rax), %edi         #  166   0xccbce  3      OPC=movl_r32_m32      
  cmpl $0xffffffff, %edi        #  167   0xccbd1  6      OPC=cmpl_r32_imm32    
  nop                           #  168   0xccbd7  1      OPC=nop               
  nop                           #  169   0xccbd8  1      OPC=nop               
  nop                           #  170   0xccbd9  1      OPC=nop               
  je .L_ccbe4                   #  171   0xccbda  2      OPC=je_label          
  cmpl %ecx, %edi               #  172   0xccbdc  2      OPC=cmpl_r32_r32      
  je .L_ccbe4                   #  173   0xccbde  2      OPC=je_label          
  leaq 0x8c9b2(%rip), %rcx      #  174   0xccbe0  7      OPC=leaq_r64_m16      
  movl $0x694, %edx             #  175   0xccbe7  5      OPC=movl_r32_imm32    
  leaq 0x904e0(%rip), %rsi      #  176   0xccbec  7      OPC=leaq_r64_m16      
  leaq 0x904f9(%rip), %rdi      #  177   0xccbf3  7      OPC=leaq_r64_m16      
  callq .__assert_fail          #  178   0xccbfa  5      OPC=callq_label       
.L_ccbe4:                       #        0xccbff  0      OPC=<label>           
  movl %ecx, 0x2c(%rax)         #  179   0xccbff  3      OPC=movl_m32_r32      
  jmpq .L_ccc34                 #  180   0xccc02  2      OPC=jmpq_label        
.L_ccbe9:                       #        0xccc04  0      OPC=<label>           
  movl 0xc(%rsp), %edi          #  181   0xccc04  4      OPC=movl_r32_m32      
  cmpl $0xffffffff, %edi        #  182   0xccc08  6      OPC=cmpl_r32_imm32    
  nop                           #  183   0xccc0e  1      OPC=nop               
  nop                           #  184   0xccc0f  1      OPC=nop               
  nop                           #  185   0xccc10  1      OPC=nop               
  je .L_ccc34                   #  186   0xccc11  2      OPC=je_label          
  movslq %edx, %rax             #  187   0xccc13  3      OPC=movslq_r64_r32    
  leaq (%rax,%rax,2), %rax      #  188   0xccc16  4      OPC=leaq_r64_m16      
  shlq $0x4, %rax               #  189   0xccc1a  4      OPC=shlq_r64_imm8     
  addq (%r12), %rax             #  190   0xccc1e  4      OPC=addq_r64_m64      
  cmpl %edi, 0x28(%rax)         #  191   0xccc22  3      OPC=cmpl_m32_r32      
  jne .L_ccc34                  #  192   0xccc25  2      OPC=jne_label         
  movl 0x2c(%rax), %edi         #  193   0xccc27  3      OPC=movl_r32_m32      
  cmpl $0xffffffff, %edi        #  194   0xccc2a  6      OPC=cmpl_r32_imm32    
  nop                           #  195   0xccc30  1      OPC=nop               
  nop                           #  196   0xccc31  1      OPC=nop               
  nop                           #  197   0xccc32  1      OPC=nop               
  je .L_ccc31                   #  198   0xccc33  2      OPC=je_label          
  cmpl %esi, %edi               #  199   0xccc35  2      OPC=cmpl_r32_r32      
  je .L_ccc31                   #  200   0xccc37  2      OPC=je_label          
  leaq 0x8c965(%rip), %rcx      #  201   0xccc39  7      OPC=leaq_r64_m16      
  movl $0x69a, %edx             #  202   0xccc40  5      OPC=movl_r32_imm32    
  leaq 0x90493(%rip), %rsi      #  203   0xccc45  7      OPC=leaq_r64_m16      
  leaq 0x904f4(%rip), %rdi      #  204   0xccc4c  7      OPC=leaq_r64_m16      
  callq .__assert_fail          #  205   0xccc53  5      OPC=callq_label       
.L_ccc31:                       #        0xccc58  0      OPC=<label>           
  movl %esi, 0x2c(%rax)         #  206   0xccc58  3      OPC=movl_m32_r32      
.L_ccc34:                       #        0xccc5b  0      OPC=<label>           
  addl $0x1, %edx               #  207   0xccc5b  3      OPC=addl_r32_imm8     
  cmpl %edx, 0x8(%r12)          #  208   0xccc5e  5      OPC=cmpl_m32_r32      
  jg .L_ccb9e                   #  209   0xccc63  6      OPC=jg_label_1        
.L_ccc42:                       #        0xccc69  0      OPC=<label>           
  cmpl $0x0, 0x28(%rsp)         #  210   0xccc69  5      OPC=cmpl_m32_imm8     
  je .L_ccc52                   #  211   0xccc6e  2      OPC=je_label          
  cmpl $0x0, 0x2c(%rsp)         #  212   0xccc70  5      OPC=cmpl_m32_imm8     
  jne .L_ccc6f                  #  213   0xccc75  2      OPC=jne_label         
  jmpq .L_ccc5b                 #  214   0xccc77  2      OPC=jmpq_label        
.L_ccc52:                       #        0xccc79  0      OPC=<label>           
  cmpl $0x0, 0x2c(%rsp)         #  215   0xccc79  5      OPC=cmpl_m32_imm8     
  je .L_ccc6f                   #  216   0xccc7e  2      OPC=je_label          
  jmpq .L_ccc65                 #  217   0xccc80  2      OPC=jmpq_label        
.L_ccc5b:                       #        0xccc82  0      OPC=<label>           
  movl $0xffffffff, %eax        #  218   0xccc82  6      OPC=movl_r32_imm32_1  
  jmpq .L_ccee9                 #  219   0xccc88  5      OPC=jmpq_label_1      
.L_ccc65:                       #        0xccc8d  0      OPC=<label>           
  movl $0x1, %eax               #  220   0xccc8d  5      OPC=movl_r32_imm32    
  jmpq .L_ccee9                 #  221   0xccc92  5      OPC=jmpq_label_1      
.L_ccc6f:                       #        0xccc97  0      OPC=<label>           
  movl 0x8(%rsp), %eax          #  222   0xccc97  4      OPC=movl_r32_m32      
  movl 0x4(%rsp), %edx          #  223   0xccc9b  4      OPC=movl_r32_m32      
  cmpl %edx, %eax               #  224   0xccc9f  2      OPC=cmpl_r32_r32      
  jl .L_cce72                   #  225   0xccca1  6      OPC=jl_label_1        
  jg .L_cce79                   #  226   0xccca7  6      OPC=jg_label_1        
  cmpb $0x0, 0x25(%rbp)         #  227   0xcccad  4      OPC=cmpb_m8_imm8      
  je .L_cce11                   #  228   0xcccb1  6      OPC=je_label_1        
  movq (%rbp), %rdx             #  229   0xcccb7  4      OPC=movq_r64_m64      
  movl 0x4(%rdx), %eax          #  230   0xcccbb  3      OPC=movl_r32_m32      
  movq (%rbx), %r12             #  231   0xcccbe  3      OPC=movq_r64_m64      
  cmpl 0x4(%r12), %eax          #  232   0xcccc1  5      OPC=cmpl_r32_m32      
  jne .L_cce11                  #  233   0xcccc6  6      OPC=jne_label_1       
  cmpl $0x2, %eax               #  234   0xccccc  3      OPC=cmpl_r32_imm8     
  jne .L_ccd59                  #  235   0xccccf  6      OPC=jne_label_1       
  cmpw $0x2, 0x8(%rbp)          #  236   0xcccd5  5      OPC=cmpw_m16_imm8     
  je .L_cccd3                   #  237   0xcccda  2      OPC=je_label          
  leaq 0x8c8c3(%rip), %rcx      #  238   0xcccdc  7      OPC=leaq_r64_m16      
  movl $0x6b7, %edx             #  239   0xccce3  5      OPC=movl_r32_imm32    
  leaq 0x903f1(%rip), %rsi      #  240   0xccce8  7      OPC=leaq_r64_m16      
  leaq 0x9049a(%rip), %rdi      #  241   0xcccef  7      OPC=leaq_r64_m16      
  callq .__assert_fail          #  242   0xcccf6  5      OPC=callq_label       
.L_cccd3:                       #        0xcccfb  0      OPC=<label>           
  cmpw $0x2, 0x8(%rbx)          #  243   0xcccfb  5      OPC=cmpw_m16_imm8     
  je .L_cccf9                   #  244   0xccd00  2      OPC=je_label          
  leaq 0x8c89d(%rip), %rcx      #  245   0xccd02  7      OPC=leaq_r64_m16      
  movl $0x6b8, %edx             #  246   0xccd09  5      OPC=movl_r32_imm32    
  leaq 0x903cb(%rip), %rsi      #  247   0xccd0e  7      OPC=leaq_r64_m16      
  leaq 0x9049c(%rip), %rdi      #  248   0xccd15  7      OPC=leaq_r64_m16      
  callq .__assert_fail          #  249   0xccd1c  5      OPC=callq_label       
.L_cccf9:                       #        0xccd21  0      OPC=<label>           
  movq 0x18(%rdx), %rax         #  250   0xccd21  4      OPC=movq_r64_m64      
  movl 0xc(%rbp), %edi          #  251   0xccd25  3      OPC=movl_r32_m32      
  xorl 0x4(%rax), %edi          #  252   0xccd28  3      OPC=xorl_r32_m32      
  bswap %edi                    #  253   0xccd2b  2      OPC=bswap_r32         
  movl $0x20, %ecx              #  254   0xccd2d  5      OPC=movl_r32_imm32    
  subb 0x27(%rbp), %cl          #  255   0xccd32  3      OPC=subb_r8_m8        
  movl $0xffffffff, %eax        #  256   0xccd35  6      OPC=movl_r32_imm32_1  
  shll %cl, %eax                #  257   0xccd3b  2      OPC=shll_r32_cl       
  movl $0x0, %ebp               #  258   0xccd3d  5      OPC=movl_r32_imm32    
  testl %eax, %edi              #  259   0xccd42  2      OPC=testl_r32_r32     
  jne .L_ccd24                  #  260   0xccd44  2      OPC=jne_label         
  callq .fls                    #  261   0xccd46  5      OPC=callq_label       
  movl %eax, %ebp               #  262   0xccd4b  2      OPC=movl_r32_r32      
.L_ccd24:                       #        0xccd4d  0      OPC=<label>           
  movq 0x18(%r12), %rax         #  263   0xccd4d  5      OPC=movq_r64_m64      
  movl 0xc(%rbx), %edi          #  264   0xccd52  3      OPC=movl_r32_m32      
  xorl 0x4(%rax), %edi          #  265   0xccd55  3      OPC=xorl_r32_m32      
  bswap %edi                    #  266   0xccd58  2      OPC=bswap_r32         
  movl $0x20, %ecx              #  267   0xccd5a  5      OPC=movl_r32_imm32    
  subb 0x27(%rbx), %cl          #  268   0xccd5f  3      OPC=subb_r8_m8        
  movl $0xffffffff, %eax        #  269   0xccd62  6      OPC=movl_r32_imm32_1  
  shll %cl, %eax                #  270   0xccd68  2      OPC=shll_r32_cl       
  movl $0x0, %edx               #  271   0xccd6a  5      OPC=movl_r32_imm32    
  testl %eax, %edi              #  272   0xccd6f  2      OPC=testl_r32_r32     
  jne .L_cce02                  #  273   0xccd71  6      OPC=jne_label_1       
  callq .fls                    #  274   0xccd77  5      OPC=callq_label       
  movl %eax, %edx               #  275   0xccd7c  2      OPC=movl_r32_r32      
  jmpq .L_cce02                 #  276   0xccd7e  5      OPC=jmpq_label_1      
.L_ccd59:                       #        0xccd83  0      OPC=<label>           
  cmpl $0xa, %eax               #  277   0xccd83  3      OPC=cmpl_r32_imm8     
  jne .L_cce11                  #  278   0xccd86  6      OPC=jne_label_1       
  cmpw $0xa, 0x8(%rbp)          #  279   0xccd8c  5      OPC=cmpw_m16_imm8     
  je .L_ccd88                   #  280   0xccd91  2      OPC=je_label          
  leaq 0x8c80e(%rip), %rcx      #  281   0xccd93  7      OPC=leaq_r64_m16      
  movl $0x6d6, %edx             #  282   0xccd9a  5      OPC=movl_r32_imm32    
  leaq 0x9033c(%rip), %rsi      #  283   0xccd9f  7      OPC=leaq_r64_m16      
  leaq 0x90435(%rip), %rdi      #  284   0xccda6  7      OPC=leaq_r64_m16      
  callq .__assert_fail          #  285   0xccdad  5      OPC=callq_label       
.L_ccd88:                       #        0xccdb2  0      OPC=<label>           
  cmpw $0xa, 0x8(%rbx)          #  286   0xccdb2  5      OPC=cmpw_m16_imm8     
  je .L_ccdae                   #  287   0xccdb7  2      OPC=je_label          
  leaq 0x8c7e8(%rip), %rcx      #  288   0xccdb9  7      OPC=leaq_r64_m16      
  movl $0x6d7, %edx             #  289   0xccdc0  5      OPC=movl_r32_imm32    
  leaq 0x90316(%rip), %rsi      #  290   0xccdc5  7      OPC=leaq_r64_m16      
  leaq 0x90437(%rip), %rdi      #  291   0xccdcc  7      OPC=leaq_r64_m16      
  callq .__assert_fail          #  292   0xccdd3  5      OPC=callq_label       
.L_ccdae:                       #        0xccdd8  0      OPC=<label>           
  movq 0x18(%rdx), %rdx         #  293   0xccdd8  4      OPC=movq_r64_m64      
  movq 0x18(%r12), %r15         #  294   0xccddc  5      OPC=movq_r64_m64      
  movl 0x10(%rbp), %eax         #  295   0xccde1  3      OPC=movl_r32_m32      
  cmpl %eax, 0x8(%rdx)          #  296   0xccde4  3      OPC=cmpl_m32_r32      
  jne .L_ccead                  #  297   0xccde7  6      OPC=jne_label_1       
  movl 0x10(%rbx), %eax         #  298   0xccded  3      OPC=movl_r32_m32      
  cmpl %eax, 0x8(%r15)          #  299   0xccdf0  4      OPC=cmpl_m32_r32      
  jne .L_cceb5                  #  300   0xccdf4  6      OPC=jne_label_1       
  movl $0x0, %eax               #  301   0xccdfa  5      OPC=movl_r32_imm32    
.L_ccdd5:                       #        0xccdff  0      OPC=<label>           
  leal 0x1(%rax), %r12d         #  302   0xccdff  4      OPC=leal_r32_m16      
  movl 0x14(%rbp,%rax,4), %ecx  #  303   0xcce03  4      OPC=movl_r32_m32      
  cmpl %ecx, 0xc(%rdx,%rax,4)   #  304   0xcce07  4      OPC=cmpl_m32_r32      
  jne .L_ccebb                  #  305   0xcce0b  6      OPC=jne_label_1       
  movl 0x14(%rbx,%rax,4), %ecx  #  306   0xcce11  4      OPC=movl_r32_m32      
  cmpl %ecx, 0xc(%r15,%rax,4)   #  307   0xcce15  5      OPC=cmpl_m32_r32      
  jne .L_ccebb                  #  308   0xcce1a  6      OPC=jne_label_1       
  addq $0x1, %rax               #  309   0xcce20  4      OPC=addq_r64_imm8     
  cmpq $0x3, %rax               #  310   0xcce24  4      OPC=cmpq_r64_imm8     
  jne .L_ccdd5                  #  311   0xcce28  2      OPC=jne_label         
  jmpq .L_cce11                 #  312   0xcce2a  2      OPC=jmpq_label        
.L_cce02:                       #        0xcce2c  0      OPC=<label>           
  cmpl %edx, %ebp               #  313   0xcce2c  2      OPC=cmpl_r32_r32      
  jg .L_cce80                   #  314   0xcce2e  2      OPC=jg_label          
  movl $0x1, %eax               #  315   0xcce30  5      OPC=movl_r32_imm32    
  jl .L_ccee9                   #  316   0xcce35  6      OPC=jl_label_1        
.L_cce11:                       #        0xcce3b  0      OPC=<label>           
  cmpq %r14, %r13               #  317   0xcce3b  3      OPC=cmpq_r64_r64      
  sbbl %eax, %eax               #  318   0xcce3e  2      OPC=sbbl_r32_r32      
  orl $0x1, %eax                #  319   0xcce40  3      OPC=orl_r32_imm8      
  jmpq .L_ccee9                 #  320   0xcce43  5      OPC=jmpq_label_1      
.L_cce1e:                       #        0xcce48  0      OPC=<label>           
  movl $0x1, %eax               #  321   0xcce48  5      OPC=movl_r32_imm32    
  jmpq .L_ccee9                 #  322   0xcce4d  5      OPC=jmpq_label_1      
.L_cce28:                       #        0xcce52  0      OPC=<label>           
  movl $0xffffffff, %eax        #  323   0xcce52  6      OPC=movl_r32_imm32_1  
  jmpq .L_ccee9                 #  324   0xcce58  5      OPC=jmpq_label_1      
.L_cce32:                       #        0xcce5d  0      OPC=<label>           
  movl $0xffffffff, %eax        #  325   0xcce5d  6      OPC=movl_r32_imm32_1  
  jmpq .L_ccee9                 #  326   0xcce63  5      OPC=jmpq_label_1      
.L_cce3c:                       #        0xcce68  0      OPC=<label>           
  movl $0x1, %eax               #  327   0xcce68  5      OPC=movl_r32_imm32    
  jmpq .L_ccee9                 #  328   0xcce6d  5      OPC=jmpq_label_1      
.L_cce46:                       #        0xcce72  0      OPC=<label>           
  movl $0x1, %eax               #  329   0xcce72  5      OPC=movl_r32_imm32    
  jmpq .L_ccee9                 #  330   0xcce77  5      OPC=jmpq_label_1      
.L_cce50:                       #        0xcce7c  0      OPC=<label>           
  movl $0xffffffff, %eax        #  331   0xcce7c  6      OPC=movl_r32_imm32_1  
  jmpq .L_ccee9                 #  332   0xcce82  5      OPC=jmpq_label_1      
.L_cce5a:                       #        0xcce87  0      OPC=<label>           
  movl $0xffffffff, %eax        #  333   0xcce87  6      OPC=movl_r32_imm32_1  
  jmpq .L_ccee9                 #  334   0xcce8d  5      OPC=jmpq_label_1      
.L_cce64:                       #        0xcce92  0      OPC=<label>           
  movl $0xffffffff, %eax        #  335   0xcce92  6      OPC=movl_r32_imm32_1  
  jmpq .L_ccee9                 #  336   0xcce98  2      OPC=jmpq_label        
.L_cce6b:                       #        0xcce9a  0      OPC=<label>           
  movl $0x1, %eax               #  337   0xcce9a  5      OPC=movl_r32_imm32    
  jmpq .L_ccee9                 #  338   0xcce9f  2      OPC=jmpq_label        
.L_cce72:                       #        0xccea1  0      OPC=<label>           
  movl $0xffffffff, %eax        #  339   0xccea1  6      OPC=movl_r32_imm32_1  
  jmpq .L_ccee9                 #  340   0xccea7  2      OPC=jmpq_label        
.L_cce79:                       #        0xccea9  0      OPC=<label>           
  movl $0x1, %eax               #  341   0xccea9  5      OPC=movl_r32_imm32    
  jmpq .L_ccee9                 #  342   0xcceae  2      OPC=jmpq_label        
.L_cce80:                       #        0xcceb0  0      OPC=<label>           
  movl $0xffffffff, %eax        #  343   0xcceb0  6      OPC=movl_r32_imm32_1  
  jmpq .L_ccee9                 #  344   0xcceb6  2      OPC=jmpq_label        
.L_cce87:                       #        0xcceb8  0      OPC=<label>           
  movq (%rbp), %rax             #  345   0xcceb8  4      OPC=movq_r64_m64      
  movq 0x18(%rax), %rdi         #  346   0xccebc  4      OPC=movq_r64_m64      
  callq .get_scope              #  347   0xccec0  5      OPC=callq_label       
  movl %eax, 0x8(%rsp)          #  348   0xccec5  4      OPC=movl_m32_r32      
  movq (%rbx), %rax             #  349   0xccec9  3      OPC=movq_r64_m64      
  movq 0x18(%rax), %rdi         #  350   0xccecc  4      OPC=movq_r64_m64      
  callq .get_scope              #  351   0xcced0  5      OPC=callq_label       
  movl %eax, 0x4(%rsp)          #  352   0xcced5  4      OPC=movl_m32_r32      
  jmpq .L_ccabd                 #  353   0xcced9  5      OPC=jmpq_label_1      
.L_ccead:                       #        0xccede  0      OPC=<label>           
  movl $0x0, %r12d              #  354   0xccede  6      OPC=movl_r32_imm32    
  jmpq .L_ccebb                 #  355   0xccee4  2      OPC=jmpq_label        
.L_cceb5:                       #        0xccee6  0      OPC=<label>           
  movl $0x0, %r12d              #  356   0xccee6  6      OPC=movl_r32_imm32    
.L_ccebb:                       #        0xcceec  0      OPC=<label>           
  movslq %r12d, %r12            #  357   0xcceec  3      OPC=movslq_r64_r32    
  movl 0x10(%rbp,%r12,4), %edi  #  358   0xcceef  5      OPC=movl_r32_m32      
  xorl 0x8(%rdx,%r12,4), %edi   #  359   0xccef4  5      OPC=xorl_r32_m32      
  bswap %edi                    #  360   0xccef9  2      OPC=bswap_r32         
  callq .fls                    #  361   0xccefb  5      OPC=callq_label       
  movl %eax, %ebp               #  362   0xccf00  2      OPC=movl_r32_r32      
  movl 0x10(%rbx,%r12,4), %edi  #  363   0xccf02  5      OPC=movl_r32_m32      
  xorl 0x8(%r15,%r12,4), %edi   #  364   0xccf07  5      OPC=xorl_r32_m32      
  bswap %edi                    #  365   0xccf0c  2      OPC=bswap_r32         
  callq .fls                    #  366   0xccf0e  5      OPC=callq_label       
  movl %eax, %edx               #  367   0xccf13  2      OPC=movl_r32_r32      
  jmpq .L_cce02                 #  368   0xccf15  5      OPC=jmpq_label_1      
.L_ccee9:                       #        0xccf1a  0      OPC=<label>           
  addq $0x38, %rsp              #  369   0xccf1a  4      OPC=addq_r64_imm8     
  popq %rbx                     #  370   0xccf1e  1      OPC=popq_r64_1        
  popq %rbp                     #  371   0xccf1f  1      OPC=popq_r64_1        
  popq %r12                     #  372   0xccf20  2      OPC=popq_r64_1        
  popq %r13                     #  373   0xccf22  2      OPC=popq_r64_1        
  popq %r14                     #  374   0xccf24  2      OPC=popq_r64_1        
  popq %r15                     #  375   0xccf26  2      OPC=popq_r64_1        
  retq                          #  376   0xccf28  1      OPC=retq              
                                                                               
.size rfc3484_sort, .-rfc3484_sort
