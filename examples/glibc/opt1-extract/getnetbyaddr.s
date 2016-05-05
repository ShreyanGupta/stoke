  .text
  .globl getnetbyaddr
  .type getnetbyaddr, @function

#! file-offset 0xef8c0
#! rip-offset  0xef8c0
#! capacity    417 bytes

# Text                                #  Line  RIP      Bytes  Opcode                
.getnetbyaddr:                        #        0xef8c0  0      OPC=<label>           
  pushq %r15                          #  1     0xef8c0  2      OPC=pushq_r64_1       
  pushq %r14                          #  2     0xef8c2  2      OPC=pushq_r64_1       
  pushq %r13                          #  3     0xef8c4  2      OPC=pushq_r64_1       
  pushq %r12                          #  4     0xef8c6  2      OPC=pushq_r64_1       
  pushq %rbp                          #  5     0xef8c8  1      OPC=pushq_r64_1       
  pushq %rbx                          #  6     0xef8c9  1      OPC=pushq_r64_1       
  subq $0x18, %rsp                    #  7     0xef8ca  4      OPC=subq_r64_imm8     
  movl %edi, %ebp                     #  8     0xef8ce  2      OPC=movl_r32_r32      
  movl %esi, %r12d                    #  9     0xef8d0  3      OPC=movl_r32_r32      
  movl $0x0, 0x4(%rsp)                #  10    0xef8d3  8      OPC=movl_m32_imm32    
  movl $0x1, %esi                     #  11    0xef8db  5      OPC=movl_r32_imm32    
  movl $0x0, %eax                     #  12    0xef8e0  5      OPC=movl_r32_imm32    
  cmpl $0x0, 0x2a1314(%rip)           #  13    0xef8e5  7      OPC=cmpl_m32_imm8     
  je .L_ef8fa                         #  14    0xef8ec  2      OPC=je_label          
  lock                                #  15    0xef8ee  1      OPC=lock              
  cmpxchgl %esi, 0x29fe3a(%rip)       #  16    0xef8ef  7      OPC=cmpxchgl_m32_r32  
  nop                                 #  17    0xef8f6  1      OPC=nop               
  jne .L_ef903                        #  18    0xef8f7  2      OPC=jne_label         
  jmpq .L_ef91d                       #  19    0xef8f9  2      OPC=jmpq_label        
.L_ef8fa:                             #        0xef8fb  0      OPC=<label>           
  cmpxchgl %esi, 0x29fe2f(%rip)       #  20    0xef8fb  7      OPC=cmpxchgl_m32_r32  
  je .L_ef91d                         #  21    0xef902  2      OPC=je_label          
.L_ef903:                             #        0xef904  0      OPC=<label>           
  leaq 0x29fe26(%rip), %rdi           #  22    0xef904  7      OPC=leaq_r64_m16      
  subq $0x80, %rsp                    #  23    0xef90b  7      OPC=subq_r64_imm32    
  callq .__lll_lock_wait_private      #  24    0xef912  5      OPC=callq_label       
  addq $0x80, %rsp                    #  25    0xef917  7      OPC=addq_r64_imm32    
.L_ef91d:                             #        0xef91e  0      OPC=<label>           
  movq 0x29d044(%rip), %rcx           #  26    0xef91e  7      OPC=movq_r64_m64      
  testq %rcx, %rcx                    #  27    0xef925  3      OPC=testq_r64_r64     
  jne .L_ef99f                        #  28    0xef928  2      OPC=jne_label         
  movq $0x400, 0x29fdf4(%rip)         #  29    0xef92a  11     OPC=movq_m64_imm32    
  movl $0x400, %edi                   #  30    0xef935  5      OPC=movl_r32_imm32    
  callq .memalign_plt                 #  31    0xef93a  5      OPC=callq_label       
  movq %rax, 0x29d023(%rip)           #  32    0xef93f  7      OPC=movq_m64_r64      
  movq %rax, %rcx                     #  33    0xef946  3      OPC=movq_r64_r64      
  testq %rax, %rax                    #  34    0xef949  3      OPC=testq_r64_r64     
  jne .L_ef99f                        #  35    0xef94c  2      OPC=jne_label         
  jmpq .L_ef9eb                       #  36    0xef94e  5      OPC=jmpq_label_1      
.L_ef952:                             #        0xef953  0      OPC=<label>           
  movq 0x29fdcf(%rip), %rax           #  37    0xef953  7      OPC=movq_r64_m64      
  leaq (%rax,%rax,1), %rsi            #  38    0xef95a  4      OPC=leaq_r64_m16      
  movq %rsi, 0x29fdc4(%rip)           #  39    0xef95e  7      OPC=movq_m64_r64      
  movq 0x29cffd(%rip), %rbx           #  40    0xef965  7      OPC=movq_r64_m64      
  movq %rbx, %rdi                     #  41    0xef96c  3      OPC=movq_r64_r64      
  callq .__tls_get_addr_plt           #  42    0xef96f  5      OPC=callq_label       
  testq %rax, %rax                    #  43    0xef974  3      OPC=testq_r64_r64     
  jne .L_efa3e                        #  44    0xef977  6      OPC=jne_label_1       
  movq %rbx, %rdi                     #  45    0xef97d  3      OPC=movq_r64_r64      
  callq .L_1f8d0                      #  46    0xef980  5      OPC=callq_label       
  movq 0x29b4f5(%rip), %rax           #  47    0xef985  7      OPC=movq_r64_m64      
  movl $0xc, (%rax)                   #  48    0xef98c  6      OPC=movl_m32_imm32    
  nop                                 #  49    0xef992  1      OPC=nop               
  movq $0x0, 0x29cfcb(%rip)           #  50    0xef993  11     OPC=movq_m64_imm32    
  jmpq .L_ef9eb                       #  51    0xef99e  2      OPC=jmpq_label        
.L_ef99f:                             #        0xef9a0  0      OPC=<label>           
  leaq 0x8(%rsp), %r15                #  52    0xef9a0  5      OPC=leaq_r64_m16      
  leaq 0x4(%rsp), %r14                #  53    0xef9a5  5      OPC=leaq_r64_m16      
  leaq 0x29fd60(%rip), %r13           #  54    0xef9aa  7      OPC=leaq_r64_m16      
.L_ef9b0:                             #        0xef9b1  0      OPC=<label>           
  subq $0x8, %rsp                     #  55    0xef9b1  4      OPC=subq_r64_imm8     
  pushq %r14                          #  56    0xef9b5  2      OPC=pushq_r64_1       
  movq %r15, %r9                      #  57    0xef9b7  3      OPC=movq_r64_r64      
  movq 0x29fd68(%rip), %r8            #  58    0xef9ba  7      OPC=movq_r64_m64      
  movq %r13, %rdx                     #  59    0xef9c1  3      OPC=movq_r64_r64      
  movl %r12d, %esi                    #  60    0xef9c4  3      OPC=movl_r32_r32      
  movl %ebp, %edi                     #  61    0xef9c7  2      OPC=movl_r32_r32      
  callq .getnetbyaddr_r__GLIBC_2_2_5  #  62    0xef9c9  5      OPC=callq_label       
  addq $0x10, %rsp                    #  63    0xef9ce  4      OPC=addq_r64_imm8     
  cmpl $0x22, %eax                    #  64    0xef9d2  3      OPC=cmpl_r32_imm8     
  jne .L_ef9e1                        #  65    0xef9d5  2      OPC=jne_label         
  cmpl $0xffffffff, 0x4(%rsp)         #  66    0xef9d7  8      OPC=cmpl_m32_imm32    
  nop                                 #  67    0xef9df  1      OPC=nop               
  nop                                 #  68    0xef9e0  1      OPC=nop               
  nop                                 #  69    0xef9e1  1      OPC=nop               
  nop                                 #  70    0xef9e2  1      OPC=nop               
  nop                                 #  71    0xef9e3  1      OPC=nop               
  je .L_ef952                         #  72    0xef9e4  6      OPC=je_label_1        
.L_ef9e1:                             #        0xef9ea  0      OPC=<label>           
  cmpq $0x0, 0x29cf7f(%rip)           #  73    0xef9ea  8      OPC=cmpq_m64_imm8     
  jne .L_ef9f4                        #  74    0xef9f2  2      OPC=jne_label         
.L_ef9eb:                             #        0xef9f4  0      OPC=<label>           
  movq $0x0, 0x8(%rsp)                #  75    0xef9f4  9      OPC=movq_m64_imm32    
.L_ef9f4:                             #        0xef9fd  0      OPC=<label>           
  cmpl $0x0, 0x2a1205(%rip)           #  76    0xef9fd  7      OPC=cmpl_m32_imm8     
  je .L_efa08                         #  77    0xefa04  2      OPC=je_label          
  lock                                #  78    0xefa06  1      OPC=lock              
  decl 0x29fd2c(%rip)                 #  79    0xefa07  6      OPC=decl_m32          
  nop                                 #  80    0xefa0d  1      OPC=nop               
  jne .L_efa10                        #  81    0xefa0e  2      OPC=jne_label         
  jmpq .L_efa2a                       #  82    0xefa10  2      OPC=jmpq_label        
.L_efa08:                             #        0xefa12  0      OPC=<label>           
  decl 0x29fd22(%rip)                 #  83    0xefa12  6      OPC=decl_m32          
  je .L_efa2a                         #  84    0xefa18  2      OPC=je_label          
.L_efa10:                             #        0xefa1a  0      OPC=<label>           
  leaq 0x29fd19(%rip), %rdi           #  85    0xefa1a  7      OPC=leaq_r64_m16      
  subq $0x80, %rsp                    #  86    0xefa21  7      OPC=subq_r64_imm32    
  callq .__lll_unlock_wake_private    #  87    0xefa28  5      OPC=callq_label       
  addq $0x80, %rsp                    #  88    0xefa2d  7      OPC=addq_r64_imm32    
.L_efa2a:                             #        0xefa34  0      OPC=<label>           
  movl 0x4(%rsp), %eax                #  89    0xefa34  4      OPC=movl_r32_m32      
  testl %eax, %eax                    #  90    0xefa38  2      OPC=testl_r32_r32     
  je .L_efa4d                         #  91    0xefa3a  2      OPC=je_label          
  movq 0x29b437(%rip), %rdx           #  92    0xefa3c  7      OPC=movq_r64_m64      
  movl %eax, (%rdx)                   #  93    0xefa43  2      OPC=movl_m32_r32      
  nop                                 #  94    0xefa45  1      OPC=nop               
  jmpq .L_efa4d                       #  95    0xefa46  2      OPC=jmpq_label        
.L_efa3e:                             #        0xefa48  0      OPC=<label>           
  movq %rax, 0x29cf23(%rip)           #  96    0xefa48  7      OPC=movq_m64_r64      
  movq %rax, %rcx                     #  97    0xefa4f  3      OPC=movq_r64_r64      
  jmpq .L_ef9b0                       #  98    0xefa52  5      OPC=jmpq_label_1      
.L_efa4d:                             #        0xefa57  0      OPC=<label>           
  movq 0x8(%rsp), %rax                #  99    0xefa57  5      OPC=movq_r64_m64      
  addq $0x18, %rsp                    #  100   0xefa5c  4      OPC=addq_r64_imm8     
  popq %rbx                           #  101   0xefa60  1      OPC=popq_r64_1        
  popq %rbp                           #  102   0xefa61  1      OPC=popq_r64_1        
  popq %r12                           #  103   0xefa62  2      OPC=popq_r64_1        
  popq %r13                           #  104   0xefa64  2      OPC=popq_r64_1        
  popq %r14                           #  105   0xefa66  2      OPC=popq_r64_1        
  popq %r15                           #  106   0xefa68  2      OPC=popq_r64_1        
  retq                                #  107   0xefa6a  1      OPC=retq              
                                                                                     
.size getnetbyaddr, .-getnetbyaddr
