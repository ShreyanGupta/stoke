  .text
  .globl getnetbyname
  .type getnetbyname, @function

#! file-offset 0xf8ca0
#! rip-offset  0xf8ca0
#! capacity    448 bytes

# Text                                #  Line  RIP      Bytes  Opcode                
.getnetbyname:                        #        0xf8ca0  0      OPC=<label>           
  pushq %r14                          #  1     0xf8ca0  2      OPC=pushq_r64_1       
  pushq %r13                          #  2     0xf8ca2  2      OPC=pushq_r64_1       
  movl $0x1, %esi                     #  3     0xf8ca4  5      OPC=movl_r32_imm32    
  pushq %r12                          #  4     0xf8ca9  2      OPC=pushq_r64_1       
  pushq %rbp                          #  5     0xf8cab  1      OPC=pushq_r64_1       
  xorl %eax, %eax                     #  6     0xf8cac  2      OPC=xorl_r32_r32      
  pushq %rbx                          #  7     0xf8cae  1      OPC=pushq_r64_1       
  movq %rdi, %rbp                     #  8     0xf8caf  3      OPC=movq_r64_r64      
  subq $0x10, %rsp                    #  9     0xf8cb2  4      OPC=subq_r64_imm8     
  movl $0x0, 0x4(%rsp)                #  10    0xf8cb6  8      OPC=movl_m32_imm32    
  cmpl $0x0, 0x2a7f3b(%rip)           #  11    0xf8cbe  7      OPC=cmpl_m32_imm8     
  je .L_f8cd3                         #  12    0xf8cc5  2      OPC=je_label          
  lock                                #  13    0xf8cc7  1      OPC=lock              
  cmpxchgl %esi, 0x2a6aa1(%rip)       #  14    0xf8cc8  7      OPC=cmpxchgl_m32_r32  
  nop                                 #  15    0xf8ccf  1      OPC=nop               
  jne .L_f8cdc                        #  16    0xf8cd0  2      OPC=jne_label         
  jmpq .L_f8cf6                       #  17    0xf8cd2  2      OPC=jmpq_label        
.L_f8cd3:                             #        0xf8cd4  0      OPC=<label>           
  cmpxchgl %esi, 0x2a6a96(%rip)       #  18    0xf8cd4  7      OPC=cmpxchgl_m32_r32  
  je .L_f8cf6                         #  19    0xf8cdb  2      OPC=je_label          
.L_f8cdc:                             #        0xf8cdd  0      OPC=<label>           
  leaq 0x2a6a8d(%rip), %rdi           #  20    0xf8cdd  7      OPC=leaq_r64_m16      
  subq $0x80, %rsp                    #  21    0xf8ce4  7      OPC=subq_r64_imm32    
  callq .__lll_lock_wait_private      #  22    0xf8ceb  5      OPC=callq_label       
  addq $0x80, %rsp                    #  23    0xf8cf0  7      OPC=addq_r64_imm32    
.L_f8cf6:                             #        0xf8cf7  0      OPC=<label>           
  movq 0x2a3c73(%rip), %rdx           #  24    0xf8cf7  7      OPC=movq_r64_m64      
  movq 0x2a6a64(%rip), %rbx           #  25    0xf8cfe  7      OPC=movq_r64_m64      
  testq %rdx, %rdx                    #  26    0xf8d05  3      OPC=testq_r64_r64     
  je .L_f8e20                         #  27    0xf8d08  6      OPC=je_label_1        
.L_f8d0d:                             #        0xf8d0e  0      OPC=<label>           
  leaq 0x4(%rsp), %r13                #  28    0xf8d0e  5      OPC=leaq_r64_m16      
  leaq 0x8(%rsp), %r12                #  29    0xf8d13  5      OPC=leaq_r64_m16      
  jmpq .L_f8d2a                       #  30    0xf8d18  2      OPC=jmpq_label        
  nop                                 #  31    0xf8d1a  1      OPC=nop               
  nop                                 #  32    0xf8d1b  1      OPC=nop               
  nop                                 #  33    0xf8d1c  1      OPC=nop               
  nop                                 #  34    0xf8d1d  1      OPC=nop               
  nop                                 #  35    0xf8d1e  1      OPC=nop               
  nop                                 #  36    0xf8d1f  1      OPC=nop               
  nop                                 #  37    0xf8d20  1      OPC=nop               
.L_f8d20:                             #        0xf8d21  0      OPC=<label>           
  movq %rax, 0x2a3c49(%rip)           #  38    0xf8d21  7      OPC=movq_m64_r64      
  movq %rax, %rdx                     #  39    0xf8d28  3      OPC=movq_r64_r64      
.L_f8d2a:                             #        0xf8d2b  0      OPC=<label>           
  leaq 0x2a6a1f(%rip), %rsi           #  40    0xf8d2b  7      OPC=leaq_r64_m16      
  movq %r13, %r9                      #  41    0xf8d32  3      OPC=movq_r64_r64      
  movq %r12, %r8                      #  42    0xf8d35  3      OPC=movq_r64_r64      
  movq %rbx, %rcx                     #  43    0xf8d38  3      OPC=movq_r64_r64      
  movq %rbp, %rdi                     #  44    0xf8d3b  3      OPC=movq_r64_r64      
  callq .getnetbyname_r__GLIBC_2_2_5  #  45    0xf8d3e  5      OPC=callq_label       
  cmpl $0x22, %eax                    #  46    0xf8d43  3      OPC=cmpl_r32_imm8     
  jne .L_f8e10                        #  47    0xf8d46  6      OPC=jne_label_1       
  cmpl $0xffffffff, 0x4(%rsp)         #  48    0xf8d4c  8      OPC=cmpl_m32_imm32    
  nop                                 #  49    0xf8d54  1      OPC=nop               
  nop                                 #  50    0xf8d55  1      OPC=nop               
  nop                                 #  51    0xf8d56  1      OPC=nop               
  nop                                 #  52    0xf8d57  1      OPC=nop               
  nop                                 #  53    0xf8d58  1      OPC=nop               
  jne .L_f8e10                        #  54    0xf8d59  6      OPC=jne_label_1       
  movq 0x2a6a0b(%rip), %rax           #  55    0xf8d5f  7      OPC=movq_r64_m64      
  movq 0x2a3c0c(%rip), %r14           #  56    0xf8d66  7      OPC=movq_r64_m64      
  leaq (%rax,%rax,1), %rbx            #  57    0xf8d6d  4      OPC=leaq_r64_m16      
  movq %r14, %rdi                     #  58    0xf8d71  3      OPC=movq_r64_r64      
  movq %rbx, %rsi                     #  59    0xf8d74  3      OPC=movq_r64_r64      
  movq %rbx, 0x2a69f3(%rip)           #  60    0xf8d77  7      OPC=movq_m64_r64      
  callq .__tls_get_addr_plt           #  61    0xf8d7e  5      OPC=callq_label       
  testq %rax, %rax                    #  62    0xf8d83  3      OPC=testq_r64_r64     
  jne .L_f8d20                        #  63    0xf8d86  2      OPC=jne_label         
  movq %r14, %rdi                     #  64    0xf8d88  3      OPC=movq_r64_r64      
  callq .L_1f8c0                      #  65    0xf8d8b  5      OPC=callq_label       
  movq 0x2a20f2(%rip), %rax           #  66    0xf8d90  7      OPC=movq_r64_m64      
  movq $0x0, 0x2a3bd7(%rip)           #  67    0xf8d97  11     OPC=movq_m64_imm32    
  movl $0xc, (%rax)                   #  68    0xf8da2  6      OPC=movl_m32_imm32    
  nop                                 #  69    0xf8da8  1      OPC=nop               
.L_f8da0:                             #        0xf8da9  0      OPC=<label>           
  movq $0x0, 0x8(%rsp)                #  70    0xf8da9  9      OPC=movq_m64_imm32    
  nop                                 #  71    0xf8db2  1      OPC=nop               
  nop                                 #  72    0xf8db3  1      OPC=nop               
  nop                                 #  73    0xf8db4  1      OPC=nop               
  nop                                 #  74    0xf8db5  1      OPC=nop               
  nop                                 #  75    0xf8db6  1      OPC=nop               
  nop                                 #  76    0xf8db7  1      OPC=nop               
  nop                                 #  77    0xf8db8  1      OPC=nop               
.L_f8db0:                             #        0xf8db9  0      OPC=<label>           
  cmpl $0x0, 0x2a7e49(%rip)           #  78    0xf8db9  7      OPC=cmpl_m32_imm8     
  je .L_f8dc4                         #  79    0xf8dc0  2      OPC=je_label          
  lock                                #  80    0xf8dc2  1      OPC=lock              
  decl 0x2a69b0(%rip)                 #  81    0xf8dc3  6      OPC=decl_m32          
  nop                                 #  82    0xf8dc9  1      OPC=nop               
  jne .L_f8dcc                        #  83    0xf8dca  2      OPC=jne_label         
  jmpq .L_f8de6                       #  84    0xf8dcc  2      OPC=jmpq_label        
.L_f8dc4:                             #        0xf8dce  0      OPC=<label>           
  decl 0x2a69a6(%rip)                 #  85    0xf8dce  6      OPC=decl_m32          
  je .L_f8de6                         #  86    0xf8dd4  2      OPC=je_label          
.L_f8dcc:                             #        0xf8dd6  0      OPC=<label>           
  leaq 0x2a699d(%rip), %rdi           #  87    0xf8dd6  7      OPC=leaq_r64_m16      
  subq $0x80, %rsp                    #  88    0xf8ddd  7      OPC=subq_r64_imm32    
  callq .__lll_unlock_wake_private    #  89    0xf8de4  5      OPC=callq_label       
  addq $0x80, %rsp                    #  90    0xf8de9  7      OPC=addq_r64_imm32    
.L_f8de6:                             #        0xf8df0  0      OPC=<label>           
  movl 0x4(%rsp), %eax                #  91    0xf8df0  4      OPC=movl_r32_m32      
  testl %eax, %eax                    #  92    0xf8df4  2      OPC=testl_r32_r32     
  je .L_f8df8                         #  93    0xf8df6  2      OPC=je_label          
  movq 0x2a207b(%rip), %rdx           #  94    0xf8df8  7      OPC=movq_r64_m64      
  movl %eax, (%rdx)                   #  95    0xf8dff  2      OPC=movl_m32_r32      
  nop                                 #  96    0xf8e01  1      OPC=nop               
.L_f8df8:                             #        0xf8e02  0      OPC=<label>           
  movq 0x8(%rsp), %rax                #  97    0xf8e02  5      OPC=movq_r64_m64      
  addq $0x10, %rsp                    #  98    0xf8e07  4      OPC=addq_r64_imm8     
  popq %rbx                           #  99    0xf8e0b  1      OPC=popq_r64_1        
  popq %rbp                           #  100   0xf8e0c  1      OPC=popq_r64_1        
  popq %r12                           #  101   0xf8e0d  2      OPC=popq_r64_1        
  popq %r13                           #  102   0xf8e0f  2      OPC=popq_r64_1        
  popq %r14                           #  103   0xf8e11  2      OPC=popq_r64_1        
  retq                                #  104   0xf8e13  1      OPC=retq              
  nop                                 #  105   0xf8e14  1      OPC=nop               
  nop                                 #  106   0xf8e15  1      OPC=nop               
  nop                                 #  107   0xf8e16  1      OPC=nop               
  nop                                 #  108   0xf8e17  1      OPC=nop               
  nop                                 #  109   0xf8e18  1      OPC=nop               
  nop                                 #  110   0xf8e19  1      OPC=nop               
.L_f8e10:                             #        0xf8e1a  0      OPC=<label>           
  cmpq $0x0, 0x2a3b58(%rip)           #  111   0xf8e1a  8      OPC=cmpq_m64_imm8     
  jne .L_f8db0                        #  112   0xf8e22  2      OPC=jne_label         
  jmpq .L_f8da0                       #  113   0xf8e24  2      OPC=jmpq_label        
  nop                                 #  114   0xf8e26  1      OPC=nop               
  nop                                 #  115   0xf8e27  1      OPC=nop               
  nop                                 #  116   0xf8e28  1      OPC=nop               
  nop                                 #  117   0xf8e29  1      OPC=nop               
.L_f8e20:                             #        0xf8e2a  0      OPC=<label>           
  movl $0x400, %edi                   #  118   0xf8e2a  5      OPC=movl_r32_imm32    
  movq $0x400, 0x2a6938(%rip)         #  119   0xf8e2f  11     OPC=movq_m64_imm32    
  movl $0x400, %ebx                   #  120   0xf8e3a  5      OPC=movl_r32_imm32    
  callq .memalign_plt                 #  121   0xf8e3f  5      OPC=callq_label       
  testq %rax, %rax                    #  122   0xf8e44  3      OPC=testq_r64_r64     
  movq %rax, 0x2a3b2c(%rip)           #  123   0xf8e47  7      OPC=movq_m64_r64      
  movq %rax, %rdx                     #  124   0xf8e4e  3      OPC=movq_r64_r64      
  jne .L_f8d0d                        #  125   0xf8e51  6      OPC=jne_label_1       
  jmpq .L_f8da0                       #  126   0xf8e57  5      OPC=jmpq_label_1      
  nop                                 #  127   0xf8e5c  1      OPC=nop               
  nop                                 #  128   0xf8e5d  1      OPC=nop               
  nop                                 #  129   0xf8e5e  1      OPC=nop               
  nop                                 #  130   0xf8e5f  1      OPC=nop               
  nop                                 #  131   0xf8e60  1      OPC=nop               
  nop                                 #  132   0xf8e61  1      OPC=nop               
  nop                                 #  133   0xf8e62  1      OPC=nop               
  nop                                 #  134   0xf8e63  1      OPC=nop               
  nop                                 #  135   0xf8e64  1      OPC=nop               
  nop                                 #  136   0xf8e65  1      OPC=nop               
  nop                                 #  137   0xf8e66  1      OPC=nop               
  nop                                 #  138   0xf8e67  1      OPC=nop               
  nop                                 #  139   0xf8e68  1      OPC=nop               
  nop                                 #  140   0xf8e69  1      OPC=nop               
                                                                                     
.size getnetbyname, .-getnetbyname
